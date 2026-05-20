"""
ml/model.py — SwarmClassifier: LSTM-based 3-class swarm threat classifier.

Input : (batch, seq_len=20, features=4)  — centroid [x, y, vx, vy] per timestep
Output: (batch, 3)                        — raw logits for classes 0/1/2

Architecture:
  LSTM  : input_size=4, hidden_size=64, num_layers=2, batch_first=True, dropout=0.3
  Head  : Linear(64→32) → ReLU → Dropout(0.3) → Linear(32→3)

Run from project root:
    python ml/model.py
"""

import sys
import pathlib

if __name__ == "__main__":
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn


class SwarmClassifier(nn.Module):
    """
    LSTM classifier for 3-class swarm threat recognition.

    Args:
        input_size:  number of features per timestep (default 4: x, y, vx, vy)
        hidden_size: LSTM hidden units (default 64)
        num_layers:  stacked LSTM layers (default 2)
        n_classes:   output classes (default 3)
        dropout:     dropout rate applied between LSTM layers and in the head (default 0.3)
    """

    def __init__(
        self,
        input_size: int = 4,
        hidden_size: int = 64,
        num_layers: int = 2,
        n_classes: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_size)
        Returns:
            logits: (batch, n_classes)
        """
        # out: (batch, seq_len, hidden_size) — take the last timestep
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


# ── Smoke test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    torch.manual_seed(0)

    model = SwarmClassifier()
    model.eval()

    batch = torch.randn(8, 20, 4)
    with torch.no_grad():
        logits = model(batch)

    n_params = sum(p.numel() for p in model.parameters())

    print(f"Input  shape : {tuple(batch.shape)}")
    print(f"Output shape : {tuple(logits.shape)}")
    print(f"Parameters   : {n_params:,}")
    print()
    print(model)
    print()
    assert logits.shape == (8, 3), f"Unexpected output shape: {logits.shape}"
    print("Smoke test passed.")
