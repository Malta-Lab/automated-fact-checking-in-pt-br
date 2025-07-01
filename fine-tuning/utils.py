# utils.py
import sys
import json
import os
import random

import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt

from datasets import AveritecDataset, LiarDataset
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
    cohen_kappa_score
)
from collections import Counter
from torch.utils.tensorboard import SummaryWriter

def get_dataset(path: str,
                tokenizer,
                split: str,
                max_length: int,
                seed: int = None,
                val_ratio: float = None):
    """
    Factory to load either AveritecDataset or LiarDataset based on 'liar' in the path.
    """
    lower = path.lower()
    if "liar" in lower:
        data_dir = path if os.path.isdir(path) else os.path.dirname(path)
        return LiarDataset(
            path=data_dir,
            tokenizer=tokenizer,
            split=split,
            max_length=max_length
        )
    elif "averitec" in lower:
        return AveritecDataset(
            path=path,
            tokenizer=tokenizer,
            split=split,
            max_length=max_length,
            val_ratio=val_ratio,
            seed=seed
        )
    else:
        raise ValueError(f"Could not infer dataset type from path: {path}")

def load_config():
    if len(sys.argv) < 2:
        print("Usage: python main.py <config_file>"); sys.exit(1)
    config_file = sys.argv[1]
    try:
        with open(config_file, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Config file {config_file} not found."); sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error decoding JSON from {config_file}."); sys.exit(1)

def set_seed(seed: int = 42):
    """
    Set random seed for Python, NumPy, PyTorch (CPU & CUDA),
    plus enforce deterministic CuDNN.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

def save_confusion_matrix(all_labels, all_preds,
                        class2idx, idx2class,
                        output_dir, dataset_type):
    """
    Compute & plot confusion matrix for LIAR or Averitec.
    """
    # pick label order
    if dataset_type == "liar":
        label_names   = ["pants-fire","false","barely-true",
                        "half-true","mostly-true","true"]
        label_indices = [class2idx[l] for l in label_names]
    else:
        label_indices = sorted(idx2class.keys())
        label_names   = [idx2class[i] for i in label_indices]

    cm = confusion_matrix(all_labels, all_preds, labels=label_indices)

    fig, ax = plt.subplots(figsize=(8,6))
    cax = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    fig.colorbar(cax, ax=ax)
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i,j] > thresh else "black"
            ax.text(j, i, f"{cm[i,j]:d}",
                    ha="center", va="center",
                    color=color, fontsize=10)
    ax.set_xticks(np.arange(len(label_names)))
    ax.set_yticks(np.arange(len(label_names)))
    ax.set_xticklabels(label_names, rotation=45, ha='right')
    ax.set_yticklabels(label_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)
    print(f"Confusion matrix saved to: {cm_path}")

def save_validation_metadata(metadata: dict, ckpt_path: str):
    """
    Dump the given metadata dict to a .yaml alongside the .bin checkpoint.
    """
    yaml_path = ckpt_path.replace(".bin", ".yaml")
    os.makedirs(os.path.dirname(yaml_path), exist_ok=True)
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(metadata, f, sort_keys=False)
    print(f"Validation metadata saved to: {yaml_path}")

def finalize_test(writer: SummaryWriter,
                output_dir: str,
                test_loss: float,
                test_metrics: dict,
                all_labels: list,
                all_preds: list,
                class2idx: dict,
                idx2class: dict,
                dataset_type: str):
    """
    1) Log scalars & markdown table
    2) Per-class P/R/F1 + support
    3) Confusion matrix
    4) (LIAR only) QWK + ordinal errors + per-class error
    5) Write single test_metrics.yaml
    """
    # ─── per-class P/R/F1 ────────────────────────────────────────────────
    if dataset_type == "liar":
        class_order   = ["pants-fire","false","barely-true",
                        "half-true","mostly-true","true"]
        label_indices = [class2idx[l] for l in class_order]
    else:
        label_indices = sorted(idx2class.keys())
        class_order   = [idx2class[i] for i in label_indices]

    p, r, f1, support = precision_recall_fscore_support(
        all_labels, all_preds,
        labels=label_indices,
        zero_division=0
    )
    per_class = {
        class_order[i]: {
            "precision": float(p[i]),
            "recall":    float(r[i]),
            "f1":        float(f1[i]),
            "support":   int(support[i])
        } for i in range(len(class_order))
    }

    # ─── TensorBoard scalars & text ──────────────────────────────────────
    writer.add_scalar("Test/Loss",      test_loss,               0)
    writer.add_scalar("Test/Accuracy",  test_metrics["accuracy"],0)
    writer.add_scalar("Test/Precision", test_metrics["precision"],0)
    writer.add_scalar("Test/Recall",    test_metrics["recall"],   0)
    writer.add_scalar("Test/F1",        test_metrics["f1"],       0)

    md = "| Metric    | Value  |\n|-----------|--------|\n"
    md += f"| Loss      | {test_loss:.4f} |\n"
    for k in ["accuracy","precision","recall","f1"]:
        md += f"| {k.capitalize():9s} | {test_metrics[k]:.4f} |\n"
    writer.add_text("Test Metrics", md, global_step=0)

    # ─── confusion matrix ────────────────────────────────────────────────
    save_confusion_matrix(
        all_labels, all_preds,
        class2idx, idx2class,
        output_dir, dataset_type
    )

    # ─── LIAR-only QWK + ordinal errors ─────────────────────────────────
    extra = {}
    if dataset_type == "liar":
        ord_map   = {lbl:i for i,lbl in enumerate(class_order)}
        true_ord  = np.array([ord_map[idx2class[t]] for t in all_labels])
        pred_ord  = np.array([ord_map[idx2class[p]] for p in all_preds])
        errors    = np.abs(pred_ord - true_ord)
        mean_err   = float(errors.mean())
        median_err = float(np.median(errors))
        mode_err   = int(Counter(errors.tolist()).most_common(1)[0][0])
        qwk        = cohen_kappa_score(true_ord, pred_ord, weights="quadratic")

        # per-class error breakdown
        per_class_error = {}
        for lbl in class_order:
            idx     = ord_map[lbl]
            errs_i  = errors[ true_ord == idx ]
            if len(errs_i):
                m   = float(errs_i.mean())
                med = float(np.median(errs_i))
                md  = int(Counter(errs_i.tolist()).most_common(1)[0][0])
            else:
                m = med = md = None
            per_class_error[lbl] = {
                "mean_error":   m,
                "median_error": med,
                "mode_error":   md
            }

        writer.add_scalar("Test/QWK",        qwk,      0)
        writer.add_scalar("Test/MeanError",   mean_err, 0)
        writer.add_scalar("Test/MedianError", median_err,0)
        writer.add_scalar("Test/ModeError",   mode_err, 0)

        extra = {
            "mean_error":               mean_err,
            "median_error":             median_err,
            "mode_error":               mode_err,
            "per_class_error":          per_class_error,
            "quadratic_weighted_kappa": float(qwk)
        }

    # ─── write YAML summary ───────────────────────────────────────────────
    summary = {
        "test_loss":      float(test_loss),
        "test_accuracy":  float(test_metrics["accuracy"]),
        "test_precision": float(test_metrics["precision"]),
        "test_recall":    float(test_metrics["recall"]),
        "test_f1":        float(test_metrics["f1"]),
        "per_class":      per_class,
    }
    summary.update(extra)

    yaml_path = os.path.join(output_dir, "test_metrics.yaml")
    with open(yaml_path, "w", encoding="utf-8") as yf:
        yaml.safe_dump(summary, yf, sort_keys=False)
    print(f"Test metrics saved to YAML: {yaml_path}")

    writer.close()