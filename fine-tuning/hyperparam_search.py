import os
import json
import optuna
import torch

from utils import load_config, set_seed
from model import load_model_and_tokenizer
from training import train_epoch, eval_model
from datasets import AveritecDataset, LiarDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # avoid warnings

def train_and_validate(cfg, lr):
    """
    Train for one trial with learning rate = lr and return validation accuracy.
    Respects cfg['freeze'] to decide whether to freeze the backbone.
    """
    # 1) apply this trial’s lr
    cfg["learning_rate"] = lr

    # 2) reproducibility
    set_seed(cfg.get("metadata", {}).get("seed", 42))

    # 3) device
    if cfg.get("gpu_device") is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg["gpu_device"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 4) Data loading
    _, tokenizer = load_model_and_tokenizer(cfg["model_name"],cache_dir="./cache")
    if cfg.get("test_path"):
        # LIAR 3-split
        DatasetClass = LiarDataset
        train_ds = DatasetClass(
            path=cfg["train_path"],
            tokenizer=tokenizer,
            split="train",
            answer=cfg["answer"],
            binary=cfg["binary"],
            language='pt' if 'results' in cfg["train_path"] else 'en',
            max_length=cfg.get("max_length", 128)
        )
        val_ds = DatasetClass(
            path=cfg["dev_path"],
            tokenizer=tokenizer,
            split="val",
            answer=cfg["answer"],
            binary=cfg["binary"],
            language='pt' if 'results' in cfg["train_path"] else 'en',
            max_length=cfg.get("max_length", 128)
        )
    else:
        # AVERITEC 2-split + internal val
        DatasetClass = AveritecDataset
        train_ds = DatasetClass(
            path=cfg["train_path"],
            tokenizer=tokenizer,
            split="train",
            val_ratio=cfg.get("val_ratio", 0.10),
            answer=cfg["answer"],
            binary=cfg["binary"],
            language='pt' if 'results' in cfg["train_path"] else 'en',
            max_length=cfg.get("max_length", 256)
        )
        val_ds = DatasetClass(
            path=cfg["train_path"],
            tokenizer=tokenizer,
            split="val",
            val_ratio=cfg.get("val_ratio", 0.10),
            answer=cfg["answer"],
            binary=cfg["binary"],
            language='pt' if 'results' in cfg["train_path"] else 'en',
            max_length=cfg.get("max_length", 256)
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg.get("num_workers", 4)
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg.get("num_workers", 4)
    )

    # 5) Model setup
    model, _ = load_model_and_tokenizer(
        cfg["model_name"],
        cache_dir="./cache",
        num_labels=len(train_ds.class2idx),
        output_attentions=False,
        output_hidden_states=False,
    )
    model.to(device)

    # freeze backbone if requested
    if cfg.get("freeze", False):
        for param in model.base_model.parameters():
            param.requires_grad = False

    optimizer = AdamW(model.parameters(), lr=lr, eps=1e-8)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="max", factor=0.1, patience=5, threshold=1e-4
    )
    criterion = torch.nn.CrossEntropyLoss()

    # 6) Training loop (early stopping on val accuracy)
    best_val_acc = 0.0
    epochs_no_improve = 0
    for _epoch in range(cfg["num_epochs"]):
        train_epoch(model, train_loader, optimizer, criterion, device)
        _, val_acc, _metrics = eval_model(model, val_loader, criterion, device)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= cfg["early_stopping_patience"]:
                break

        scheduler.step(val_acc)

    return best_val_acc


def objective(trial):
    # load base config
    cfg = load_config()
    model_name = cfg["model_name"]
    freeze = cfg.get("freeze", False)

    # choose LR range based on model & freeze flag
    if freeze:
        # head-only fine-tuning
        if "multilingual" in model_name.lower():
            lr = trial.suggest_loguniform("learning_rate", 1e-4, 1e-3)
        else:
            # both BERT-Large and BERTimbau-Large
            lr = trial.suggest_loguniform("learning_rate", 5e-6, 5e-4)
    else:
        # full fine-tuning
        if "multilingual" in model_name.lower():
            lr = trial.suggest_loguniform("learning_rate", 5e-6, 2e-4)
        else:
            lr = trial.suggest_loguniform("learning_rate", 1e-6, 1e-4)

    # run training + validation
    val_acc = train_and_validate(cfg, lr)
    return val_acc


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10)

    print(f"Best validation Accuracy: {study.best_value:.4f}")
    print(f"Best learning rate: {study.best_params['learning_rate']}")
        
    # Save trial history for inspection
    cfg = load_config()
    exp_name = cfg.get("experiment_name", "optuna")
    csv_name = f"{exp_name}_optuna_trials.csv"
    study.trials_dataframe().to_csv(csv_name, index=False)    
    
    print(f"Trials saved to {csv_name}")

# implementing pruning, try this later

# import os
# import json
# import optuna
# import torch

# from utils import load_config, set_seed
# from model import load_model_and_tokenizer
# from training import train_epoch, eval_model
# from datasets import AveritecDataset, LiarDataset
# from torch.optim import AdamW
# from torch.optim.lr_scheduler import ReduceLROnPlateau
# from torch.utils.data import DataLoader

# os.environ["TOKENIZERS_PARALLELISM"] = "false"  # avoid warnings

# def train_and_validate(cfg, lr, trial):
#     """
#     Train for one trial with learning rate = lr and return validation accuracy.
#     Reports intermediate val_acc to Optuna for pruning.
#     """
#     cfg = cfg.copy()
#     cfg["learning_rate"] = lr
#     set_seed(cfg.get("metadata", {}).get("seed", 42))
#     if cfg.get("gpu_device") is not None:
#         os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg["gpu_device"])
#     device = "cuda" if torch.cuda.is_available() else "cpu"

#     # —— Data loading —— 
#     _, tokenizer = load_model_and_tokenizer(
#         cfg["model_name"], cache_dir=cfg.get("cache_dir")
#     )
#     if cfg.get("test_path"):
#         DatasetClass = LiarDataset
#         train_ds = DatasetClass(
#             path=cfg["train_path"],
#             tokenizer=tokenizer,
#             split="train",
#             answer=cfg["answer"],
#             binary=cfg["binary"],
#             language='pt' if 'results' in cfg["train_path"] else 'en',
#             max_length=cfg.get("max_length", 128)
#         )
#         val_ds = DatasetClass(
#             path=cfg["dev_path"],
#             tokenizer=tokenizer,
#             split="val",
#             answer=cfg["answer"],
#             binary=cfg["binary"],
#             language='pt' if 'results' in cfg["train_path"] else 'en',
#             max_length=cfg.get("max_length", 128)
#         )
#     else:
#         DatasetClass = AveritecDataset
#         train_ds = DatasetClass(
#             path=cfg["train_path"],
#             tokenizer=tokenizer,
#             split="train",
#             val_ratio=cfg.get("val_ratio", 0.10),
#             answer=cfg["answer"],
#             binary=cfg["binary"],
#             language='pt' if 'results' in cfg["train_path"] else 'en',
#             max_length=cfg.get("max_length", 256)
#         )
#         val_ds = DatasetClass(
#             path=cfg["train_path"],
#             tokenizer=tokenizer,
#             split="val",
#             val_ratio=cfg.get("val_ratio", 0.10),
#             answer=cfg["answer"],
#             binary=cfg["binary"],
#             language='pt' if 'results' in cfg["train_path"] else 'en',
#             max_length=cfg.get("max_length", 256)
#         )

#     train_loader = DataLoader(
#         train_ds, batch_size=cfg["batch_size"], shuffle=True,
#         num_workers=cfg.get("num_workers", 4)
#     )
#     val_loader = DataLoader(
#         val_ds, batch_size=cfg["batch_size"], shuffle=False,
#         num_workers=cfg.get("num_workers", 4)
#     )

#     # —— Model setup —— 
#     model, _ = load_model_and_tokenizer(
#         cfg["model_name"],
#         cache_dir=cfg.get("cache_dir"),
#         num_labels=len(train_ds.class2idx),
#         output_attentions=False,
#         output_hidden_states=False
#     )
#     model.to(device)
#     if cfg.get("freeze", False):
#         for p in model.base_model.parameters():
#             p.requires_grad = False

#     optimizer = AdamW(model.parameters(), lr=lr, eps=1e-8)
#     scheduler = ReduceLROnPlateau(
#         optimizer, mode="max", factor=0.1, patience=5, threshold=1e-4
#     )
#     criterion = torch.nn.CrossEntropyLoss()

#     # —— Training loop with pruning —— 
#     best_val_acc = 0.0
#     no_improve   = 0
#     for epoch in range(cfg["num_epochs"]):
#         train_epoch(model, train_loader, optimizer, criterion, device)
#         _, val_acc, _ = eval_model(model, val_loader, criterion, device)

#         # report to Optuna
#         trial.report(val_acc, epoch)
#         if trial.should_prune():
#             raise optuna.TrialPruned()

#         if val_acc > best_val_acc:
#             best_val_acc  = val_acc
#             no_improve    = 0
#         else:
#             no_improve += 1
#             if no_improve >= cfg["early_stopping_patience"]:
#                 break

#         scheduler.step(val_acc)

#     return best_val_acc


# def objective(trial):
#     cfg        = load_config()
#     model_name = cfg["model_name"].lower()
#     freeze     = cfg.get("freeze", False)

#     # choose LR range
#     if freeze:
#         if "multilingual" in model_name:
#             lr = trial.suggest_loguniform("learning_rate", 1e-4, 1e-3)
#         else:
#             lr = trial.suggest_loguniform("learning_rate", 5e-6, 5e-4)
#     else:
#         if "multilingual" in model_name:
#             lr = trial.suggest_loguniform("learning_rate", 5e-6, 2e-4)
#         else:
#             lr = trial.suggest_loguniform("learning_rate", 1e-6, 1e-4)

#     val_acc = train_and_validate(cfg, lr, trial)
#     return val_acc


# if __name__ == "__main__":
#     study = optuna.create_study(
#         direction="maximize",
#         pruner=optuna.pruners.MedianPruner(n_startup_trials=2, n_warmup_steps=1)
#     )
#     study.optimize(objective, n_trials=10)

#     print(f"✅ Best validation Accuracy: {study.best_value:.4f}")
#     print(f"✅ Best learning rate:      {study.best_params['learning_rate']:.2e}")

#     cfg       = load_config()
#     exp_name  = cfg.get("experiment_name", "optuna")
#     csv_name  = f"{exp_name}_optuna_trials.csv"
#     study.trials_dataframe().to_csv(csv_name, index=False)
#     print(f"Trials saved to: {csv_name}")
