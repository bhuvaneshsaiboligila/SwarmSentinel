"""
fusion/fusion_pipeline.py — Top-level sensor fusion orchestrator.

Reads all three sensors each frame and hands their outputs to TrackManager.
This file is intentionally thin — it is glue, not logic.

Call order each frame (driven externally):
    swarm.step(dt)              ← advance the physics simulation
    detections = pipeline.step()       ← sense → fuse; returns raw detection dict
    tracks = pipeline.confirmed_tracks()  ← get current confirmed track list
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

    def __init__(self, swarm, dt: float = 1 / 15, n_sensors: int = 3, use_ekf: bool = False):
        """
        Args:
            swarm:     Swarm instance — sensors hold a reference and read it each step.
            dt:        Timestep in seconds; must match the simulation DT.
            n_sensors: How many sensor modalities to activate (1=radar, 2=+optical,
                       3=+RF). Sensors beyond n_sensors return empty detections.
            use_ekf:   If True, TrackManager uses DroneEKF (5-state CTR) instead of
                       DroneKalmanFilter (4-state CV). Default False preserves existing behaviour.
        """
        # Drone.bounds is a (w, h) tuple since the rectangular-bounds fix.
        # Use it directly as the optical-sensor area so FPs stay within the canvas.
        area = swarm.drones[0].bounds if swarm.drones else (100.0, 100.0)

        self._radar        = RadarSensor(swarm)
        self._optical      = OpticalSensor(swarm, area=area)
        self._rf           = RFSensor(swarm)
        self.track_manager = TrackManager(dt=dt, use_ekf=use_ekf)
        self._n_sensors    = n_sensors

    def confirmed_tracks(self) -> list:
        """Return confirmed tracks from the most recent step()."""
        return self.track_manager.confirmed_tracks()

    def step(self) -> dict:
        """
        Run one full sense → fuse cycle.

        1. Sample active sensors from the current swarm state.
        2. Pass detections to TrackManager (predict + associate + update + prune).
        3. Return the raw sensor readings so callers can visualise them.
           Confirmed tracks are available via self.track_manager.confirmed_tracks().

        Returns:
            Dict with keys "radar", "optical", "rf" containing raw sensor output.
        """
        radar_det   = self._radar.sense()                                         # always active
        optical_det = self._optical.sense() if self._n_sensors >= 2 else []      # sensor 2
        rf_det      = self._rf.sense()      if self._n_sensors >= 3 else []      # sensor 3

        self.track_manager.step(radar_det, optical_det, rf_det)

        return {"radar": radar_det, "optical": optical_det, "rf": rf_det}


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from simulator.swarm import Swarm

    DT = 1 / 15
    swarm    = Swarm(n_drones=10, spawn_area=(100, 100), behavior="flock")
    pipeline = FusionPipeline(swarm, dt=DT)

    print("FusionPipeline smoke test — 10 drones, flock behavior, 5 steps")
    print(f"{'Step':>5}  {'Active Tracks':>14}")
    print("-" * 24)

    for step in range(5):
        swarm.step(DT)
        detections = pipeline.step()
        tracks     = pipeline.confirmed_tracks()
        print(f"{step:>5}  {len(tracks):>14}")

    assert len(tracks) > 0, "Expected at least one active track after 5 steps"
    print(f"\nDetections this frame: {sum(len(v) for v in detections.values())}")
    print(f"Confirmed tracks: {len(tracks)}")
