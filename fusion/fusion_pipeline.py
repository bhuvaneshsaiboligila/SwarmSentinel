"""
fusion/fusion_pipeline.py — Top-level sensor fusion orchestrator.

Reads all three sensors each frame and hands their outputs to TrackManager.
This file is intentionally thin — it is glue, not logic.

Call order each frame (driven externally):
    swarm.step(dt)          ← advance the physics simulation
    pipeline.step()         ← sense → fuse → return track states
"""

import sys
import pathlib

if __name__ == "__main__":
    # Allow `python fusion/fusion_pipeline.py` from the project root.
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from sensors.radar import RadarSensor
from sensors.optical import OpticalSensor
from sensors.rf import RFSensor
from fusion.track_manager import TrackManager


class FusionPipeline:
    """
    Glue layer between the swarm simulator and the track manager.

    Creates one instance of each sensor and one TrackManager.
    step() reads all sensors, feeds them to the manager, and returns
    the active track list in a flat, easy-to-consume format.
    """

    def __init__(self, swarm, dt: float = 1 / 15):
        """
        Args:
            swarm: Swarm instance — sensors hold a reference and read it each step.
            dt:    Timestep in seconds; must match the simulation DT.
        """
        # Derive canvas size from the drones' wrap boundary so optical false
        # positives are placed within the visible area (CLAUDE.md known issue #2).
        bounds = swarm.drones[0].bounds if swarm.drones else 100.0
        area = (bounds, bounds)

        self._radar   = RadarSensor(swarm)
        self._optical = OpticalSensor(swarm, area=area)
        self._rf      = RFSensor(swarm)
        self._tm      = TrackManager(dt=dt)

    def step(self) -> list:
        """
        Run one full sense → fuse cycle.

        1. Sample all three sensors from the current swarm state.
        2. Pass detections to TrackManager (predict + associate + update + prune).
        3. Flatten track states into {track_id, x, y, vx, vy} dicts.

        Returns:
            List of dicts, one per active track.
        """
        radar_det   = self._radar.sense()    # (n_detected, 4): [x, y, vx, vy]
        optical_det = self._optical.sense()  # list of {drone_id, bbox, confidence}
        rf_det      = self._rf.sense()       # (n_alive,): signal strengths

        raw_states = self._tm.step(radar_det, optical_det, rf_det)

        return [
            {
                "track_id": s["track_id"],
                "x":  s["position"][0],
                "y":  s["position"][1],
                "vx": s["velocity"][0],
                "vy": s["velocity"][1],
            }
            for s in raw_states
        ]


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from simulator.swarm import Swarm

    DT = 1 / 15
    swarm    = Swarm(n_drones=10, spawn_area=(100, 100), behavior="flock")
    pipeline = FusionPipeline(swarm, dt=DT)

    print("FusionPipeline smoke test — 10 drones, flock behavior, 5 steps")
    print(f"{'Step':>5}  {'Active Tracks':>14}")
    print("-" * 24)

    tracks = []
    for step in range(5):
        swarm.step(DT)           # advance physics
        tracks = pipeline.step() # sense + fuse
        print(f"{step:>5}  {len(tracks):>14}")

    assert len(tracks) > 0, "Expected at least one active track after 5 steps"
    print(f"\nDone. Pipeline is alive with {len(tracks)} active track(s).")
