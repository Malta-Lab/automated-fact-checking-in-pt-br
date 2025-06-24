import json
import sys
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score
import numpy as np

# ============================ CONFIGURATION ============================

INPUT_PREDICTIONS = Path(sys.argv[1])

ORIGINAL_FILES = [
    Path("./dataset/train.jsonl"),
    Path("./dataset/valid.jsonl"),
    Path("./dataset/test.jsonl")
]

ORDERED_CLASSES = [
    "pants-fire",
    "false",
    "barely-true",
    "half-true",
    "mostly-true",
    "true"
]

# =======================================================================

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def load_gold(paths):
    gold = {}
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            if path.suffix == ".jsonl":
                for line in f:
                    item = json.loads(line)
                    key = item.get("statement") or item.get("claim")
                    if key:
                        gold[key] = item["label"]
            else:
                data = json.load(f)
                for item in data:
                    key = item.get("statement") or item.get("claim")
                    if key:
                        gold[key] = item["label"]
    return gold

def load_predictions(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
        return {item.get("statement") or item.get("claim"): item["label"] for item in data}

def print_confusion_matrix(y_true, y_pred, labels):
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    print("\n📊 Confusion Matrix (True Label × Predicted Label):\n")
    header = f"{'':35}" + "".join(f"{label[:20]:>25}" for label in labels)
    print(header)
    for i, row in enumerate(matrix):
        row_label = labels[i]
        row_str = f"{row_label[:33]:35}" + "".join(f"{n:>25}" for n in row)
        print(row_str)

def print_report(y_true, y_pred, labels):
    print(f"\n📊 Overall Evaluation")
    print(f"Matched claims: {len(y_true)}")
    correct = sum(yt == yp for yt, yp in zip(y_true, y_pred))
    print(f"Exact matches: {correct} ({correct / len(y_true):.1%})\n")
    print_confusion_matrix(y_true, y_pred, labels)
    print("\n📋 Classification Report:\n")
    print(classification_report(y_true, y_pred, labels=labels, digits=3, zero_division=0))

from collections import Counter

def calculate_ordinal_error(y_true, y_pred, ordered_labels):
    """
    Compute absolute ordinal errors and return:
        - errors: list of absolute differences (including zeros)
        - y_true_ord: list of true ordinal indices
        - y_pred_ord: list of predicted ordinal indices
        - mode_error: mode of errors only over misclassified instances (errors > 0), or None
    """
    errors = []
    y_true_ord, y_pred_ord = [], []
    label_to_idx = {label: i for i, label in enumerate(ordered_labels)}
    invalid_labels = {}

    for real, pred in zip(y_true, y_pred):
        try:
            idx_real = label_to_idx[real]
            idx_pred = label_to_idx[pred]
            diff = abs(idx_real - idx_pred)
            errors.append(diff)
            y_true_ord.append(idx_real)
            y_pred_ord.append(idx_pred)
        except KeyError:
            invalid_labels.setdefault(pred, 0)
            invalid_labels[pred] += 1

    if invalid_labels:
        print("⚠️ Invalid predicted labels encountered:")
        for label, count in invalid_labels.items():
            print(f"  → '{label}': {count} occurrence(s)")

    # Compute mode only over misclassified instances
    misclassified_errors = [e for e in errors if e != 0]
    if misclassified_errors:
        mode_error = Counter(misclassified_errors).most_common(1)[0][0]
    else:
        mode_error = None

    return errors, y_true_ord, y_pred_ord, mode_error


# def calculate_ordinal_error(y_true, y_pred, ordered_labels):
#     errors = []
#     y_true_ord, y_pred_ord = [], []
#     label_to_idx = {label: i for i, label in enumerate(ordered_labels)}
#     invalid_labels = {}

#     for real, pred in zip(y_true, y_pred):
#         try:
#             idx_real = label_to_idx[real]
#             idx_pred = label_to_idx[pred]
#             errors.append(abs(idx_real - idx_pred))
#             y_true_ord.append(idx_real)
#             y_pred_ord.append(idx_pred)
#         except KeyError:
#             invalid_labels.setdefault(pred, 0)
#             invalid_labels[pred] += 1

#     if invalid_labels:
#         print("⚠️ Invalid predicted labels encountered:")
#         for label, count in invalid_labels.items():
#             print(f"  → '{label}': {count} occurrence(s)")

#     return errors, y_true_ord, y_pred_ord

# ============================ MAIN EXECUTION ============================

if __name__ == "__main__":
    gold_data = load_gold(ORIGINAL_FILES)
    pred_data = load_predictions(INPUT_PREDICTIONS)

    y_true, y_pred = [], []
    missing = 0
    for claim, label_real in gold_data.items():
        label_pred = pred_data.get(claim)
        if not label_pred:
            missing += 1
            continue
        y_true.append(label_real)
        y_pred.append(label_pred.strip().lower())

    if missing:
        print(f"⚠️ Claims not found in predictions: {missing}")

    # Classification report
    print_report(y_true, y_pred, ORDERED_CLASSES)

    # Ordinal error analysis
    ordinal_errors, y_true_ord, y_pred_ord, mode_err = calculate_ordinal_error(y_true, y_pred, ORDERED_CLASSES)
    if ordinal_errors:
        positive_errors = [e for e in ordinal_errors if e > 0]
        mode = np.bincount(positive_errors).argmax() if positive_errors else "Undefined"
        mean = np.mean(positive_errors) if positive_errors else 0
        median = np.median(positive_errors) if positive_errors else 0
        qwk = cohen_kappa_score(y_true_ord, y_pred_ord, weights="quadratic")

        print("\n📐 Ordinal Errors:")
        print(f"Mean error: {round(mean, 3)}")
        print(f"Median error: {round(median, 3)}")
        print(f"Mode error: {mode}")
        print(f"QWK (Quadratic Weighted Kappa): {round(qwk, 4)}")
        print(f"Total evaluated examples: {len(ordinal_errors)}")
