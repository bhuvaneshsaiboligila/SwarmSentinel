"""
ml/dataset.py — Synthetic swarm trajectory dataset for Phase 3 classification.

Generates fixed-length centroid trajectory sequences from the existing simulator
and radar sensor. Three classes:
  0 — individual drone  (random behavior, n=1)
  1 — swarm             (flock behavior,  n=15–25)
  2 — swarm attack      (attack behavior, n=15–25, target = arena centre)

Each sample is a sequence of T timesteps; each timestep is an 8-dim feature vector:
  [centroid_x, centroid_y, centroid_vx, centroid_vy,
   spread_x, spread_y, n_alive_norm, mean_speed]

Run from project root:
    python ml/dataset.py
"""

import sys
import pathlib

if __name__ == "__main__":
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from simulator.swarm import Swarm
from sensors.radar import RadarSensor

_AREA   = (100, 100)
_CENTER = np.array([_AREA[0] / 2.0, _AREA[1] / 2.0])
_DT     = 1 / 15


class SyntheticSwarmDataset:
    """
    Synthetic dataset of centroid trajectory sequences for 3-class swarm classification.

    Args:
        n_samples_per_class: number of sequences to generate per label (total = 3x this)
        seq_len:             timesteps per sequence (T)
        noise:               if True, centroid is derived from RadarSensor detections;
                             if False, centroid is computed from ground-truth positions
        seed:                master RNG seed for full reproducibility
        area:                arena (width, height) in metres

    Attributes:
        X: float32 numpy array, shape (N, T, 8) — feature vector per timestep
           [cx, cy, cvx, cvy, spread_x, spread_y, n_alive_norm, mean_speed]
        y: int64   numpy array, shape (N,)       — class labels 0/1/2
    """

    def __init__(
        self,
        n_samples_per_class: int = 500,
        seq_len: int = 20,
        noise: bool = True,
        seed: int = 42,
        area: tuple = _AREA,
    ):
        self.n_samples_per_class = n_samples_per_class
        self.seq_len = seq_len
        self.noise = noise
        self.area = area
        self._center = np.array([area[0] / 2.0, area[1] / 2.0])

        rng = np.random.default_rng(seed)
        # Pre-draw per-sample numpy seeds so order never affects reproducibility.
        n_total = n_samples_per_class * 3
        sample_seeds = rng.integers(0, 2**31, size=n_total)

        X_list, y_list = [], []
        for label in range(3):
            for i in range(n_samples_per_class):
                idx = label * n_samples_per_class + i
                seq = self._generate_sequence(label, int(sample_seeds[idx]), rng)
                X_list.append(seq)
                y_list.append(label)

        self.X = np.array(X_list, dtype=np.float32)   # (N, T, 8)
        self.y = np.array(y_list, dtype=np.int64)      # (N,)

    # ── sequence generation ────────────────────────────────────────────────────

    def _generate_sequence(
        self, label: int, np_seed: int, rng: np.random.Generator
    ) -> np.ndarray:
        """Simulate one scenario and return a (seq_len, 8) feature trajectory."""
        # np_seed controls Swarm/Sensor internal randomness for this sample.
        np.random.seed(np_seed)

        if label == 0:
            swarm = Swarm(n_drones=1, spawn_area=self.area, behavior="random")
        elif label == 1:
            n = int(rng.integers(15, 26))
            swarm = Swarm(n_drones=n, spawn_area=self.area, behavior="flock")
        else:  # label == 2
            n = int(rng.integers(15, 26))
            swarm = Swarm(
                n_drones=n, spawn_area=self.area,
                behavior="attack", target=self._center.copy(),
            )

        radar = RadarSensor(swarm) if self.noise else None

        seq = []
        for _ in range(self.seq_len):
            swarm.step(_DT)
            seq.append(self._features(swarm, radar))

        return np.array(seq, dtype=np.float32)  # (T, 8)

    def _features(self, swarm: Swarm, radar) -> np.ndarray:
        """Return 8-dim feature: [cx, cy, cvx, cvy, spread_x, spread_y, n_alive_norm, mean_speed].

        Centroid (first 4): from radar detections when noise=True, else ground truth.
        Structural features (last 4): always from ground truth — spread and n_alive are
        not reliably recoverable from noisy radar alone, but they powerfully discriminate classes:
          individual  → spread≈0, n_alive_norm≈0.04
          flock       → spread>0, mean_speed low (drones average out)
          attack      → spread shrinking, mean_speed rising toward target
        """
        alive = [d for d in swarm.drones if d.alive]
        n_alive = len(alive)

        if alive:
            positions  = np.array([d.position for d in alive])   # (n, 2)
            velocities = np.array([d.velocity for d in alive])   # (n, 2)
            spread_x   = float(positions[:, 0].std()) if n_alive > 1 else 0.0
            spread_y   = float(positions[:, 1].std()) if n_alive > 1 else 0.0
            mean_speed = float(np.sqrt((velocities ** 2).sum(axis=1)).mean())
        else:
            positions = velocities = None
            spread_x = spread_y = mean_speed = 0.0

        n_alive_norm = n_alive / 25.0

        # Noisy centroid from radar; fall back to ground truth when radar misses everything
        if radar is not None:
            det = radar.sense()   # (n_detected, 4): [x, y, vx, vy]
            if det.shape[0] > 0:
                centroid = det.mean(axis=0)
            elif positions is not None:
                centroid = np.concatenate([positions.mean(axis=0), velocities.mean(axis=0)])
            else:
                centroid = np.zeros(4)
        elif positions is not None:
            centroid = np.concatenate([positions.mean(axis=0), velocities.mean(axis=0)])
        else:
            centroid = np.zeros(4)

        return np.array(
            [centroid[0], centroid[1], centroid[2], centroid[3],
             spread_x, spread_y, n_alive_norm, mean_speed],
            dtype=np.float32,
        )

    # ── Dataset interface ──────────────────────────────────────────────────────

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, idx):
        import torch
        return (
            torch.from_numpy(self.X[idx].copy()),                      # float32, (T, 4)
            torch.tensor(int(self.y[idx]), dtype=torch.long),          # int64
        )


# ── Smoke test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating 30 samples (10 per class) …")
    ds = SyntheticSwarmDataset(n_samples_per_class=10, seq_len=20, noise=True, seed=0)

    print(f"  X shape : {ds.X.shape}")          # expect (30, 20, 8)
    print(f"  y shape : {ds.y.shape}")           # expect (30,)
    print(f"  X dtype : {ds.X.dtype}")
    print(f"  y dtype : {ds.y.dtype}")
    print(f"  Classes : {sorted(set(ds.y.tolist()))}  "
          f"counts={[(ds.y == c).sum() for c in range(3)]}")

    print()
    label_names = {0: "individual", 1: "swarm", 2: "attack"}
    header = "  {:>12}  {:>6} {:>6} {:>6} {:>6}  {:>6} {:>6}  {:>5} {:>5}".format(
        "class", "cx", "cy", "cvx", "cvy", "sx", "sy", "n", "spd"
    )
    print(header)
    for cls in range(3):
        row = ds.X[cls * 10, 0]
        print(f"  {cls} ({label_names[cls]:<10})  "
              f"{row[0]:6.2f} {row[1]:6.2f} {row[2]:6.2f} {row[3]:6.2f}  "
              f"{row[4]:6.2f} {row[5]:6.2f}  {row[6]:.3f} {row[7]:5.2f}")

    print()
    x0, y0 = ds[0]
    print(f"  __getitem__(0) → x type={type(x0).__name__} shape={x0.shape}, "
          f"y type={type(y0).__name__} value={y0}")

    assert ds.X.shape == (30, 20, 8), f"Unexpected X shape: {ds.X.shape}"
    assert ds.y.shape == (30,),       f"Unexpected y shape: {ds.y.shape}"
    assert set(ds.y.tolist()) == {0, 1, 2}
    print()
    print("Smoke test passed.")
