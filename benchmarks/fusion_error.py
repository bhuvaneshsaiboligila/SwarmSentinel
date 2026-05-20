"""
benchmarks/fusion_error.py — Fusion accuracy benchmark.

Three configurations matching ROADMAP.md Phase 2 milestone (line 87):
  Config 1: radar only              (1 sensor, baseline)
  Config 2: radar + optical         (2 sensors, position fusion benefit)
  Config 3: radar + optical + RF    (3 sensors, RF as spawn-confidence gate)

RF carries no position information, so Config 3 differs from Config 2 only in
that TrackManager uses the RF drone-count estimate to gate new-track spawning:
a new track is suppressed when the number of live tracks already meets or
exceeds the count of positively-detected RF signals.

Run from project root:
    python benchmarks/fusion_error.py
"""

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from simulator.swarm import Swarm
from fusion.fusion_pipeline import FusionPipeline

N_STEPS  = 200
N_DRONES = 25
AREA     = (100, 100)
DT       = 1 / 15


def benchmark_config(n_sensors: int, n_steps: int = N_STEPS) -> tuple:
    """
    Run one benchmark configuration and return (rmse, avg_tracks_confirmed).

    Args:
        n_sensors: 1 = radar only, 2 = radar + optical, 3 = + RF confidence gate.
        n_steps:   Number of simulation steps (default: 200).

    Returns:
        (rmse, avg_tracks): position RMSE in metres; mean confirmed-track count.
    """
    # Fixed seed so all configs see the same swarm trajectory.
    np.random.seed(42)

    swarm    = Swarm(n_drones=N_DRONES, spawn_area=AREA, behavior="flock")
    pipeline = FusionPipeline(swarm, dt=DT, n_sensors=n_sensors)

    sq_errors       = []
    tracks_per_step = []

    for _ in range(n_steps):
        swarm.step(DT)

        # Ground-truth positions of alive drones — shape (n_alive, 2)
        gt = np.array([d.position.copy() for d in swarm.drones if d.alive])

        # Sense + fuse; confirmed tracks available after step().
        pipeline.step()
        tracks = pipeline.track_manager.confirmed_tracks()
        tracks_per_step.append(len(tracks))

        if not tracks:
            continue

        est = np.array([[t["position"][0], t["position"][1]] for t in tracks])  # (n_tracks, 2)

        # Hungarian optimal assignment — minimises total track-to-drone distance.
        cost = cdist(est, gt)
        row_idx, col_idx = linear_sum_assignment(cost)

        for r, c in zip(row_idx, col_idx):
            dx = est[r, 0] - gt[c, 0]
            dy = est[r, 1] - gt[c, 1]
            sq_errors.append(dx * dx + dy * dy)

    rmse       = float(np.sqrt(np.mean(sq_errors))) if sq_errors else float("inf")
    avg_tracks = float(np.mean(tracks_per_step))    if tracks_per_step else 0.0
    return rmse, avg_tracks


if __name__ == "__main__":
    CONFIGS = [
        (1, "Radar only"),
        (2, "Radar + Optical"),
        (3, "Radar + Optical + RF"),
    ]

    print("Fusion error benchmark — Phase 2")
    print(f"  {N_DRONES} drones, flock behavior, {N_STEPS} steps, DT={DT:.4f} s")
    print()
    print(f"  {'Config':<24} {'RMSE (m)':>10} {'tracks_confirmed':>18}")
    print("  " + "-" * 54)

    results = []
    for n_sensors, label in CONFIGS:
        rmse, avg_tracks = benchmark_config(n_sensors)
        results.append((label, rmse, avg_tracks))
        print(f"  {label:<24} {rmse:>10.3f} {avg_tracks:>18.1f}")

    print()
    rmse_1 = results[0][1]
    rmse_2 = results[1][1]
    rmse_3 = results[2][1]

    if rmse_2 < rmse_1:
        print(f"  +Optical reduces RMSE by {rmse_1 - rmse_2:.3f} m over radar-only.")
    else:
        print(f"  +Optical shows no improvement ({rmse_2:.3f} m vs {rmse_1:.3f} m).")

    if rmse_3 < rmse_2:
        print(f"  +RF gate reduces RMSE by {rmse_2 - rmse_3:.3f} m over radar+optical.")
    else:
        print(f"  +RF gate shows no improvement ({rmse_3:.3f} m vs {rmse_2:.3f} m).")
