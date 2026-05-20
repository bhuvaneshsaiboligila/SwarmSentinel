"""
ml/predictor.py — Physics-based trajectory predictor for drone/swarm centroids.

No learned model — uses linear extrapolation from the last 3 observed centroid
positions in a 20-step feature sequence.  Simple and interpretable; appropriate
for short-horizon (≤10 step) predictions on the synthetic arena.

Run from project root:
    python ml/predictor.py
"""

import sys
import pathlib

if __name__ == "__main__":
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np

ARENA_BOUNDS = (0.0, 100.0)   # both axes; arena is square


class TrajectoryPredictor:
    """
    Predicts future centroid positions using linear extrapolation.

    Fits a degree-1 polynomial (line) through the last ``lookback`` centroid
    positions in a (T, 8) feature sequence and extrapolates ``horizon`` steps
    forward.  Predictions are clamped to [0, 100] on both axes.

    Args:
        horizon:  number of future timesteps to predict (default 10)
        lookback: how many trailing positions to fit the line through (default 3)
        arena:    (lo, hi) clamp range for both axes (default (0.0, 100.0))
    """

    def __init__(
        self,
        horizon: int = 10,
        lookback: int = 3,
        arena: tuple = ARENA_BOUNDS,
    ):
        self.horizon  = horizon
        self.lookback = lookback
        self.lo       = float(arena[0])
        self.hi       = float(arena[1])

    def predict(self, sequence: np.ndarray) -> np.ndarray:
        """
        Predict the next ``horizon`` centroid positions.

        Args:
            sequence: (T, 8) float array — the observed feature sequence.
                      Features 0 and 1 are centroid_x and centroid_y.
                      T must be >= lookback.

        Returns:
            np.ndarray shape (horizon, 2): predicted [x, y] for each future step,
            clamped to arena bounds.
        """
        if sequence.ndim != 2 or sequence.shape[1] < 2:
            raise ValueError(f"Expected (T, >=2) array, got {sequence.shape}")
        if sequence.shape[0] < self.lookback:
            raise ValueError(
                f"Sequence length {sequence.shape[0]} < lookback {self.lookback}"
            )

        # Extract the last `lookback` centroid positions: shape (lookback, 2)
        recent = sequence[-self.lookback:, :2].astype(float)

        # Time indices for observed points and future points
        t_obs    = np.arange(self.lookback, dtype=float)
        t_future = np.arange(self.lookback, self.lookback + self.horizon, dtype=float)

        # Fit a line through each axis independently (degree-1 polyfit)
        cx = np.polyfit(t_obs, recent[:, 0], 1)
        cy = np.polyfit(t_obs, recent[:, 1], 1)

        x_pred = np.polyval(cx, t_future)
        y_pred = np.polyval(cy, t_future)

        # Clamp to arena bounds
        x_pred = np.clip(x_pred, self.lo, self.hi)
        y_pred = np.clip(y_pred, self.lo, self.hi)

        return np.stack([x_pred, y_pred], axis=1).astype(np.float32)  # (horizon, 2)


# ── Smoke test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from ml.dataset import SyntheticSwarmDataset

    ds        = SyntheticSwarmDataset(n_samples_per_class=1, seq_len=20, noise=True, seed=7)
    predictor = TrajectoryPredictor(horizon=10, lookback=3)

    label_names = {0: "individual", 1: "swarm", 2: "attack"}

    print(f"TrajectoryPredictor smoke test — horizon=10, lookback=3")
    print()

    for cls in range(3):
        seq = ds.X[cls]            # (20, 8)  — first sample for each class
        pred = predictor.predict(seq)  # (10, 2)

        last_obs = seq[-1, :2]
        print(f"  class={cls} ({label_names[cls]:<10})")
        print(f"    last observed  : x={last_obs[0]:6.2f}  y={last_obs[1]:6.2f}")
        print(f"    predicted t+1  : x={pred[0, 0]:6.2f}  y={pred[0, 1]:6.2f}")
        print(f"    predicted t+5  : x={pred[4, 0]:6.2f}  y={pred[4, 1]:6.2f}")
        print(f"    predicted t+10 : x={pred[9, 0]:6.2f}  y={pred[9, 1]:6.2f}")
        print()

    assert pred.shape == (10, 2), f"Unexpected shape: {pred.shape}"
    assert pred.dtype == np.float32
    assert np.all(pred >= 0.0) and np.all(pred <= 100.0), "Prediction out of bounds"
    print("Smoke test passed.")
