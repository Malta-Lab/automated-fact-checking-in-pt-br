# main.py
from utils import load_config, set_seed, save_validation_metadata, finalize_test
from model import load_model_and_tokenizer
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from training import train_epoch, eval_model
from datasets import AveritecDataset, LiarDataset
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support, cohen_kappa_score

import torch
import torch.nn as nn

import os
import yaml

from torch.utils.tensorboard import SummaryWriter  

# for reproducibility of the LIAR “ordinal‐error” metrics
import numpy as np
from collections import Counter

os.environ["TOKENIZERS_PARALLELISM"] = "false"

config = load_config()

# GPU setup
gpu_id = config.get("gpu_device", None)
if gpu_id is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
else:
    raise ValueError("No GPU device specified in config.json. Stopping training.")
device = "cuda" if torch.cuda.is_available() else "cpu"

# Hyperparams & paths
model_name              = config.get("model_name", "bert-base-uncased")
num_epochs              = config.get("num_epochs", 10)
experiment_name         = config.get("experiment_name", None)
learning_rate           = config.get("learning_rate", 2e-5)
batch_size              = config.get("batch_size", 16)
early_stopping_patience = config.get("early_stopping_patience", 100)
two_layer_head          = config.get("two_layer_head", False)

cache_dir   = config.get("cache_dir", "./cache")
ckpt_dir    = config.get("ckpt_dir", "checkpoints")

train_path  = config.get("train_path")
dev_path    = config.get("dev_path")
test_path   = config.get("test_path", None)
binary      = config.get("binary", False)
answer      = config.get("answer", False)
freeze      = config.get("freeze", False)
class_weights = config.get("class_weights", False)
language      = 'pt' if 'results' in train_path else 'en'

# seed (this also sets np, torch, cuda seeds and cudnn flags)
seed = config.get("metadata", {}).get("seed", 42)
set_seed(seed)

# sanity check
if train_path is None or dev_path is None:
    raise ValueError("Please set at least 'train_path' and 'dev_path' in your config.")

# derive experiment name if needed
if experiment_name is None:
    safe_model_name  = model_name.replace('/', '-')
    experiment_name = f"{safe_model_name}_lr{learning_rate}"

runs_root    = "runs"
exp_folder   = os.path.join(runs_root, experiment_name)
exp_ckpt_dir = os.path.join(exp_folder, "checkpoints")
exp_tb_dir   = os.path.join(exp_folder, "tensorboard")
os.makedirs(exp_ckpt_dir, exist_ok=True)
os.makedirs(exp_tb_dir, exist_ok=True)

# load tokenizer
_, tokenizer = load_model_and_tokenizer(model_name)

# //// DATASET INSTANTIATION ////////////////////////////////////////////

if test_path:
    # LIAR 3-split
    train_dataset = LiarDataset(
        path=train_path,
        tokenizer=tokenizer,
        split='train',
        answer=answer,
        binary=binary,
        language=language,
        max_length=config.get("max_length", 128)
    )
    val_dataset   = LiarDataset(
        path=dev_path,
        tokenizer=tokenizer,
        split='val',
        answer=answer,
        binary=binary,
        language=language,
        max_length=config.get("max_length", 128)
    )
    test_dataset  = LiarDataset(
        path=test_path,
        tokenizer=tokenizer,
        split='test',
        answer=answer,
        binary=binary,
        language=language,
        max_length=config.get("max_length", 128)
    )
else:
    # Averitec 2-split + internal val
    train_dataset = AveritecDataset(
        path=train_path,
        tokenizer=tokenizer,
        split='train',
        val_ratio=config.get("val_ratio", 0.20),
        binary=binary,
        answer=answer,
        language=language,
        max_length=config.get("max_length", 256)
    )
    val_dataset   = AveritecDataset(
        path=train_path,
        tokenizer=tokenizer,
        split='val',
        val_ratio=config.get("val_ratio", 0.10),
        binary=binary,
        answer=answer,
        language=language,
        max_length=config.get("max_length", 256)
    )
    test_dataset  = AveritecDataset(
        path=dev_path,
        tokenizer=tokenizer,
        split='test',
        binary=binary,
        answer=answer,
        language=language,
        max_length=config.get("max_length", 256)
    )

# DataLoaders
num_workers  = config.get("num_workers", 32)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=num_workers)
test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=num_workers)

# //// MODEL SETUP ////////////////////////////////////////

model, _ = load_model_and_tokenizer(
    model_name,
    cache_dir=cache_dir,
    num_labels=len(train_dataset.class2idx),
    output_attentions=False,
    output_hidden_states=False
)
model.to(device)

if freeze:
    for p in model.base_model.parameters():
        p.requires_grad = False

if two_layer_head and model_name == "bert-base-multilingual-cased":
    hidden     = model.config.hidden_size
    num_labels = model.config.num_labels
    mlp_hidden = hidden // 2
    model.classifier = nn.Sequential(
        nn.Linear(hidden, mlp_hidden),
        nn.ReLU(),
        nn.Dropout(model.config.classifier_dropout),
        nn.Linear(mlp_hidden, num_labels)
    )
    print(f"Replaced head with two-layer MLP ({hidden}→{mlp_hidden}→{num_labels})")

# TensorBoard setup
writer = SummaryWriter(log_dir=exp_tb_dir)

optimizer = AdamW(model.parameters(), lr=learning_rate, eps=1e-8)

scheduler = ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.1,
    patience=5,
    threshold=1e-4
)

# Loss
if class_weights:
    cw = train_dataset.get_class_weights().to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=cw)
else:
    criterion = torch.nn.CrossEntropyLoss()

# train & val loops
best_val_f1       = 0.0
best_val_acc      = 0.0
epochs_no_improve = 0
patience          = early_stopping_patience

for epoch in range(num_epochs):
    actual_epochs_ran = epoch + 1
    print(f"\n------------ Epoch {actual_epochs_ran}/{num_epochs} ------------")

    # 1) Training step
    train_loss, train_acc = train_epoch(
        model, train_loader, optimizer, criterion, device
    )
    print(f"Train loss: {train_loss:.4f} | Train accuracy: {train_acc:.4f}")

    # Log training metrics
    writer.add_scalar("Loss/Train",    train_loss, epoch)
    writer.add_scalar("Accuracy/Train", train_acc, epoch)

    # 2) Validation step
    val_loss, _, val_metrics = eval_model(
        model, val_loader, criterion, device
    )
    print(
        f"Val loss: {val_loss:.4f} | "
        f"Accuracy: {val_metrics['accuracy']:.4f} | "
        f"Precision: {val_metrics['precision']:.4f} | "
        f"Recall: {val_metrics['recall']:.4f} | "
        f"F1: {val_metrics['f1']:.4f}"
    )

    # Log validation metrics
    writer.add_scalar("Loss/Validation",    val_loss,                epoch)
    writer.add_scalar("Accuracy/Validation", val_metrics["accuracy"], epoch)
    writer.add_scalar("Precision/Validation", val_metrics["precision"], epoch)
    writer.add_scalar("Recall/Validation",    val_metrics["recall"],    epoch)
    writer.add_scalar("F1/Validation",        val_metrics["f1"],        epoch)

    # 3) Early‐stopping & checkpointing:
    #    - Averitec: based on F1
    #    - LIAR:     based on Accuracy
    if test_path:
        # LIAR: early-stop on accuracy
        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc      = val_metrics["accuracy"]
            epochs_no_improve = 0
            key, best_val     = "accuracy", best_val_acc
        else:
            epochs_no_improve += 1
    else:
        # AVERITEC: early-stop on f1
        if val_metrics["f1"] > best_val_f1:
            best_val_f1        = val_metrics["f1"]
            epochs_no_improve  = 0
            key, best_val      = "f1", best_val_f1
        else:
            epochs_no_improve += 1

    # if improved, save checkpoint + metadata
    if ((test_path and val_metrics["accuracy"] == best_val_acc) or
        (not test_path and val_metrics["f1"] == best_val_f1)):
        os.makedirs(exp_ckpt_dir, exist_ok=True)
        ckpt_name = f"best_{model_name.replace('/', '-')}_lr{learning_rate}.bin"
        ckpt_path = os.path.join(exp_ckpt_dir, ckpt_name)
        torch.save(model.state_dict(), ckpt_path)
        print(f"****** Best model saved to: {ckpt_path} ({key}={best_val:.4f}) ******")

        # assemble the same metadata dict as before
        metadata = {
            "experiment_name":     experiment_name,
            "model_name":          model_name,
            "cache_dir":           cache_dir,
            "train_path":          train_path,
            "dev_path":            dev_path,
            "test_path":           test_path,
            "device":              str(device),
            "num_epochs_planned":  num_epochs,
            "num_epochs_ran":      actual_epochs_ran,
            "early_stop_patience": patience,
            "learning_rate":       learning_rate,
            "batch_size":          batch_size,
            "freeze":              freeze,
            "class_weights":       class_weights,
            "optimizer":           "AdamW",
            "scheduler":           "ReduceLROnPlateau",
            "two_layer_head":      two_layer_head,
            "answer":              answer,
            "binary":              binary,
            "ckpt_name":           ckpt_name,
            "scheduler_params": {
                "mode":      "max",
                "factor":    0.1,
                "patience":  5,
                "threshold": 1e-4
            },
            "validation_metrics_at_save": {
                "val_loss":      val_loss,
                "val_accuracy":  val_metrics["accuracy"],
                "val_precision": val_metrics["precision"],
                "val_recall":    val_metrics["recall"],
                "val_f1":        val_metrics["f1"],
            }
        }
        # now use our util to write it
        save_validation_metadata(metadata, ckpt_path)

    else:
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered! No improvement in {key} for {patience} epochs.")
            print(f"End of training {experiment_name} at epoch {actual_epochs_ran}/{num_epochs}.\n")
            break

    # 4) Step the scheduler on the same key
    scheduler.step(val_metrics[key])

# ─── Final evaluation + test-time metrics ─────────────────────────────────────

print("\n=== Final evaluation on test set ===")
test_loss, _, test_metrics = eval_model(model, test_loader, criterion, device)
print(
    f"Test loss: {test_loss:.4f} | "
    f"Accuracy: {test_metrics['accuracy']:.4f} | "
    f"Precision: {test_metrics['precision']:.4f} | "
    f"Recall: {test_metrics['recall']:.4f} | "
    f"F1: {test_metrics['f1']:.4f}"
)

# collect all preds & labels
all_preds, all_labels = [], []
model.eval()
with torch.no_grad():
    for batch in test_loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)
        outputs        = model(input_ids=input_ids, attention_mask=attention_mask)
        preds          = torch.argmax(outputs.logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# delegate full dumping to the util
dataset_type = "liar" if test_path else "averitec"
finalize_test(
    writer=writer,
    output_dir=exp_ckpt_dir,
    test_loss=test_loss,
    test_metrics=test_metrics,
    all_labels=all_labels,
    all_preds=all_preds,
    class2idx=train_dataset.class2idx,
    idx2class=train_dataset.idx2class,
    dataset_type=dataset_type
)

print(f"End of testing {experiment_name} with model {model_name}.")