"""
ml/train.py — Training loop for SwarmClassifier.

Generates a 3-class synthetic dataset, splits 80/20, trains for 30 epochs with
Adam and CrossEntropyLoss, and saves the best checkpoint to ml/checkpoints/.

Run from project root:
    python ml/train.py
"""

import sys
import pathlib

if __name__ == "__main__":
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from ml.dataset import SyntheticSwarmDataset
from ml.model import SwarmClassifier

CHECKPOINT_DIR = pathlib.Path(__file__).parent / "checkpoints"

# ── Hyperparameters ────────────────────────────────────────────────────────────
N_PER_CLASS = 300
BATCH_SIZE  = 32
EPOCHS      = 30
LR          = 1e-3
SEED        = 42


def generate_dataset(n_per_class: int = N_PER_CLASS) -> SyntheticSwarmDataset:
    """Generate a fresh SyntheticSwarmDataset with n_per_class samples per label."""
    return SyntheticSwarmDataset(
        n_samples_per_class=n_per_class, seq_len=20, noise=True, seed=SEED,
    )


def train():
    torch.manual_seed(SEED)

    # ── Dataset + split ────────────────────────────────────────────────────────
    print(f"Generating dataset ({N_PER_CLASS} samples/class × 3 classes) …")
    dataset = generate_dataset()
    n_val   = int(0.2 * len(dataset))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(SEED),
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)
    print(f"  Train samples : {n_train}  |  Val samples : {n_val}")
    print()

    # ── Model + optimiser + loss ───────────────────────────────────────────────
    model     = SwarmClassifier()
    optimiser = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    best_val_acc   = 0.0
    best_state     = None

    # ── Training loop ──────────────────────────────────────────────────────────
    print(f"{'Epoch':>5}  {'Train Loss':>10}  {'Val Loss':>8}  {'Val Acc':>8}")
    print("─" * 40)

    for epoch in range(1, EPOCHS + 1):
        # Train
        model.train()
        train_loss = 0.0
        for X_b, y_b in train_loader:
            optimiser.zero_grad()
            loss = criterion(model(X_b), y_b)
            loss.backward()
            optimiser.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # Validate
        model.eval()
        val_loss = 0.0
        correct  = 0
        with torch.no_grad():
            for X_b, y_b in val_loader:
                logits    = model(X_b)
                val_loss += criterion(logits, y_b).item()
                correct  += (logits.argmax(dim=1) == y_b).sum().item()
        val_loss /= len(val_loader)
        val_acc   = correct / n_val

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state   = {k: v.clone() for k, v in model.state_dict().items()}

        print(f"{epoch:>5}  {train_loss:>10.4f}  {val_loss:>8.4f}  {val_acc:>7.1%}")

    # ── Save best checkpoint ───────────────────────────────────────────────────
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = CHECKPOINT_DIR / "swarm_classifier.pt"
    torch.save(best_state, ckpt_path)

    print()
    print(f"Best val accuracy : {best_val_acc:.1%}")
    print(f"Checkpoint saved  → {ckpt_path}")


if __name__ == "__main__":
    train()
