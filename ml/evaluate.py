"""
ml/evaluate.py — Evaluation script for SwarmClassifier.

Loads ml/checkpoints/swarm_classifier.pt, generates a fresh held-out test set
(200 samples/class, seed=99 — never seen during training which used seed=42),
and reports:
  - Overall accuracy
  - Per-class precision, recall, F1
  - 3×3 confusion matrix (ASCII)

Saves the confusion matrix to ml/checkpoints/confusion_matrix.txt.

Run from project root:
    python ml/evaluate.py
"""

import sys
import pathlib

if __name__ == "__main__":
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch.utils.data import DataLoader

from ml.dataset import SyntheticSwarmDataset
from ml.model import SwarmClassifier

CHECKPOINT_PATH  = pathlib.Path(__file__).parent / "checkpoints" / "swarm_classifier.pt"
CONFUSION_OUT    = pathlib.Path(__file__).parent / "checkpoints" / "confusion_matrix.txt"
LABEL_NAMES      = ["individual", "swarm", "attack"]
N_PER_CLASS      = 200
TEST_SEED        = 99   # distinct from training seed (42)
BATCH_SIZE       = 64


# ── Metric helpers (no sklearn dependency) ─────────────────────────────────────

def build_confusion(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def per_class_metrics(cm: np.ndarray):
    """Return (precision, recall, f1) arrays, one value per class."""
    n = cm.shape[0]
    precision = np.zeros(n)
    recall    = np.zeros(n)
    f1        = np.zeros(n)
    for c in range(n):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        precision[c] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall[c]    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        denom        = precision[c] + recall[c]
        f1[c]        = 2 * precision[c] * recall[c] / denom if denom > 0 else 0.0
    return precision, recall, f1


def format_confusion(cm: np.ndarray, label_names: list) -> str:
    """Render the confusion matrix as a plain-text ASCII table."""
    col_w  = max(len(n) for n in label_names) + 2   # column width
    num_w  = max(col_w, 6)

    lines = []
    lines.append("Confusion Matrix  (rows = Actual, cols = Predicted)")
    lines.append("")

    # Header row
    header = " " * (col_w + 2) + "".join(f"{n:>{num_w}}" for n in label_names)
    lines.append(header)
    lines.append(" " * (col_w + 2) + "─" * (num_w * len(label_names)))

    for i, row_name in enumerate(label_names):
        row = f"{row_name:>{col_w}} │" + "".join(f"{cm[i, j]:>{num_w}}" for j in range(len(label_names)))
        lines.append(row)

    return "\n".join(lines)


# ── Main evaluation ────────────────────────────────────────────────────────────

def evaluate():
    # ── Load model ─────────────────────────────────────────────────────────────
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {CHECKPOINT_PATH}. Run ml/train.py first."
        )
    model = SwarmClassifier()
    model.load_state_dict(torch.load(CHECKPOINT_PATH, weights_only=True))
    model.eval()
    print(f"Loaded checkpoint: {CHECKPOINT_PATH}")

    # ── Generate fresh test set ────────────────────────────────────────────────
    print(f"Generating test set ({N_PER_CLASS} samples/class × 3, seed={TEST_SEED}) …")
    test_ds     = SyntheticSwarmDataset(n_samples_per_class=N_PER_CLASS, seq_len=20,
                                        noise=True, seed=TEST_SEED)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    # ── Run inference ──────────────────────────────────────────────────────────
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_b, y_b in test_loader:
            preds = model(X_b).argmax(dim=1)
            all_preds.append(preds.numpy())
            all_labels.append(y_b.numpy())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)

    # ── Metrics ────────────────────────────────────────────────────────────────
    n_classes = len(LABEL_NAMES)
    cm        = build_confusion(y_true, y_pred, n_classes)
    accuracy  = float((y_true == y_pred).mean())
    precision, recall, f1 = per_class_metrics(cm)

    macro_p  = precision.mean()
    macro_r  = recall.mean()
    macro_f1 = f1.mean()

    # ── Print results ──────────────────────────────────────────────────────────
    print()
    cm_str = format_confusion(cm, LABEL_NAMES)
    print(cm_str)

    print()
    print(f"{'Class':<12}  {'Precision':>9}  {'Recall':>6}  {'F1':>6}")
    print("─" * 40)
    for c, name in enumerate(LABEL_NAMES):
        print(f"{name:<12}  {precision[c]:>9.3f}  {recall[c]:>6.3f}  {f1[c]:>6.3f}")
    print("─" * 40)
    print(f"{'macro avg':<12}  {macro_p:>9.3f}  {macro_r:>6.3f}  {macro_f1:>6.3f}")
    print()
    print(f"Overall accuracy : {accuracy:.1%}  ({int(y_true == y_pred).sum() if False else (y_true == y_pred).sum()}/{len(y_true)})")

    # ── Save confusion matrix ──────────────────────────────────────────────────
    CONFUSION_OUT.parent.mkdir(parents=True, exist_ok=True)
    report_lines = [
        cm_str,
        "",
        f"{'Class':<12}  {'Precision':>9}  {'Recall':>6}  {'F1':>6}",
        "─" * 40,
    ]
    for c, name in enumerate(LABEL_NAMES):
        report_lines.append(f"{name:<12}  {precision[c]:>9.3f}  {recall[c]:>6.3f}  {f1[c]:>6.3f}")
    report_lines += [
        "─" * 40,
        f"{'macro avg':<12}  {macro_p:>9.3f}  {macro_r:>6.3f}  {macro_f1:>6.3f}",
        "",
        f"Overall accuracy : {accuracy:.1%}  ({(y_true == y_pred).sum()}/{len(y_true)})",
    ]
    CONFUSION_OUT.write_text("\n".join(report_lines) + "\n")
    print(f"Saved → {CONFUSION_OUT}")

    return accuracy, precision, recall, f1, cm


if __name__ == "__main__":
    evaluate()
