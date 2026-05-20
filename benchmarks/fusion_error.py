"""
benchmarks/fusion_error.py — Fusion accuracy benchmark.

Three honest configs, each measuring something real:
  Config 1: radar only,      KF tracker  (baseline)
  Config 2: radar + optical, KF tracker  (sensor fusion benefit)
  Config 3: radar + optical, EKF tracker (nonlinear motion model benefit)

RF is excluded: TrackManager ignores RF signal strength (no position info),
so including it would produce a delta driven purely by RNG noise.

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


def benchmark_config(n_sensors: int, use_ekf: bool = False, n_steps: int = N_STEPS) -> float:
    """
    Return the average position RMSE (metres) for one benchmark configuration.

    Args:
        n_sensors: 1 = radar only, 2 = radar + optical.
        use_ekf:   False = DroneKalmanFilter (4-state CV),
                   True  = DroneEKF (5-state CTR).
        n_steps:   Number of simulation steps to run (default: 200).

    Returns:
        RMSE in metres between fused track estimates and ground-truth positions,
        averaged over all matched (track, drone) pairs across all steps.
    """
    # Fixed seed per run so all configs see the same swarm trajectory.
    np.random.seed(42)

    swarm    = Swarm(n_drones=N_DRONES, spawn_area=AREA, behavior="flock")
    pipeline = FusionPipeline(swarm, dt=DT, n_sensors=n_sensors, use_ekf=use_ekf)

    sq_errors = []

    for _ in range(n_steps):
        swarm.step(DT)

        # Ground-truth positions of alive drones — shape (n_alive, 2)
        gt = np.array([d.position.copy() for d in swarm.drones if d.alive])

        # Sense + fuse; confirmed tracks available after step().
        pipeline.step()
        tracks = pipeline.track_manager.confirmed_tracks()
        if not tracks:
            continue

        est = np.array([[t["position"][0], t["position"][1]] for t in tracks])  # (n_tracks, 2)

        # Hungarian optimal assignment: minimises total track-to-drone distance.
        # linear_sum_assignment handles rectangular matrices: matches
        # min(n_tracks, n_alive) pairs and leaves the rest unmatched.
        cost = cdist(est, gt)                          # (n_tracks, n_alive)
        row_idx, col_idx = linear_sum_assignment(cost)

        for r, c in zip(row_idx, col_idx):
            dx = est[r, 0] - gt[c, 0]
            dy = est[r, 1] - gt[c, 1]
            sq_errors.append(dx * dx + dy * dy)

    return float(np.sqrt(np.mean(sq_errors))) if sq_errors else float("inf")


if __name__ == "__main__":
    CONFIGS = [
        (1, False, "Radar only (KF)"),
        (2, False, "Radar+Optical (KF)"),
        (2, True,  "Radar+Optical (EKF)"),
    ]

    print("Fusion error benchmark — Phase 2")
    print(f"  {N_DRONES} drones, flock behavior, {N_STEPS} steps, DT={DT:.4f} s")
    print()
    print(f"  {'Config':<24} {'RMSE':>8}")
    print("  " + "-" * 34)

    results = {}
    for n_sensors, use_ekf, label in CONFIGS:
        rmse = benchmark_config(n_sensors, use_ekf=use_ekf)
        results[label] = rmse
        print(f"  {label:<24} {rmse:>7.3f} m")

    print()
    kf_base   = results["Radar only (KF)"]
    kf_fused  = results["Radar+Optical (KF)"]
    ekf_fused = results["Radar+Optical (EKF)"]

    if kf_fused < kf_base:
        print(f"  +Optical reduces RMSE by {kf_base - kf_fused:.3f} m over radar-only KF.")
    else:
        print(f"  +Optical shows no improvement ({kf_fused:.3f} m vs {kf_base:.3f} m).")

    if ekf_fused < kf_fused:
        print(f"  EKF reduces RMSE by {kf_fused - ekf_fused:.3f} m over KF with same sensors.")
    else:
        print(f"  EKF shows no improvement over KF ({ekf_fused:.3f} m vs {kf_fused:.3f} m).")
