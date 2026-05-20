"""
fusion/track_manager.py — Multi-target track lifecycle manager.

A Track wraps one DroneKalmanFilter and records how long it has been alive
and how many consecutive frames it has gone without a measurement.

A TrackManager holds all active Tracks and runs one full predict-associate-
update cycle per simulation step across radar and optical sensor streams.

RF is accepted as a parameter for API completeness but not used for spatial
association — signal strength alone carries no position information.
"""

import sys
import pathlib

if __name__ == "__main__":
    # When run as `python fusion/track_manager.py`, Python puts fusion/ in sys.path,
    # not the project root. This one-liner fixes that so `fusion` is importable.
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from scipy.optimize import linear_sum_assignment
from fusion.kalman_filter import DroneKalmanFilter
from fusion.ekf import DroneEKF

# A track is pruned after this many consecutive frames with no matching detection.
MAX_MISSED = 8

# Maximum Euclidean distance (metres) to consider a detection a match for a track.
ASSOC_THRESHOLD = 15.0

# Tracks younger than this are excluded from the output list.
# Optical false positives rarely survive 3 frames without another hit, so this
# kills ghost tracks before they ever reach the caller.
MIN_AGE = 3


class Track:
    """
    One tracked drone: a Kalman filter plus lifecycle bookkeeping.

    predict() is called every step regardless of whether a detection arrives.
    update() is called only when a detection is associated to this track,
    and it resets missed_frames to zero.
    """

    def __init__(
        self,
        track_id: int,
        x: float,
        y: float,
        vx: float = 0.0,
        vy: float = 0.0,
        dt: float = 1 / 15,
        filter_cls=DroneKalmanFilter,
    ):
        self.track_id = track_id
        self.age = 0            # total steps this track has existed
        self.missed_frames = 0  # consecutive steps without a matching detection

        self._kf = filter_cls(dt=dt)
        self._kf.initialize(x=x, y=y, vx=vx, vy=vy)

    def predict(self):
        """
        Advance the filter one step using the motion model.

        Called every frame before association. We increment missed_frames here
        so that if update() is never called this step the counter rises; update()
        will reset it back to zero if a match is found.
        """
        self._kf.predict()
        self.age += 1
        self.missed_frames += 1

    def update(self, measurement: np.ndarray, mode: str):
        """
        Fuse a matched detection into the Kalman filter.

        Args:
            measurement: shape (4,) [x,y,vx,vy] for radar,
                         shape (2,) [x,y] for optical
            mode:        'radar' or 'optical'
        """
        if mode == "radar":
            self._kf.radar_update(measurement)
        elif mode == "optical":
            self._kf.optical_update(measurement)
        else:
            raise ValueError(f"Unknown mode {mode!r}. Expected 'radar' or 'optical'.")
        self.missed_frames = 0  # reset: we just heard from this drone

    def get_position(self) -> np.ndarray:
        """Return estimated position as [x, y]."""
        return self._kf.get_state()[:2]

    def get_state(self) -> np.ndarray:
        """Return full state estimate as [x, y, vx, vy]."""
        return self._kf.get_state()

    def is_dead(self) -> bool:
        """True when the track has coasted too long without a detection."""
        return self.missed_frames > MAX_MISSED

    def __repr__(self):
        s = self.get_state()
        return (
            f"Track(id={self.track_id}, age={self.age}, "
            f"missed={self.missed_frames}, "
            f"pos=[{s[0]:.1f}, {s[1]:.1f}], vel=[{s[2]:.1f}, {s[3]:.1f}])"
        )


class TrackManager:
    """
    Manages the full lifecycle of all active drone tracks.

    Each call to step() runs:
      1. Predict    — roll every track forward one timestep
      2. Associate  — globally optimal assignment via scipy linear_sum_assignment
      3. Update     — fuse matched detections; unmatched detections spawn new tracks
      4. Prune      — remove tracks that have missed too many frames
    """

    def __init__(self, dt: float = 1 / 15, use_ekf: bool = False):
        self.dt = dt
        self.use_ekf = use_ekf
        self._tracks: list = []
        self._next_id = 0
        self._last_confirmed: list = []

    @property
    def n_tracks(self) -> int:
        """Number of currently active tracks."""
        return len(self._tracks)

    def step(
        self,
        radar_detections: np.ndarray,
        optical_detections: list,
        rf_detections: np.ndarray,
    ) -> list:
        """
        Run one full predict-associate-update-prune cycle.

        Args:
            radar_detections  : (n, 4) float array [x, y, vx, vy].
                                Pass np.empty((0, 4)) when radar has no returns.
            optical_detections: list of dicts from OpticalSensor.sense():
                                {drone_id, bbox: [x, y, w, h], confidence}.
                                False positives (drone_id=None) are treated
                                identically — the manager has no ground truth.
            rf_detections     : (n,) float array of signal strengths.
                                Accepted for API symmetry; not used for spatial
                                association because signal strength has no position.

        Returns:
            List of state dicts for every active track:
            [{track_id, position, velocity, age, missed_frames}, ...]
        """
        # ── Step 1: Predict every existing track forward ───────────────────────
        for track in self._tracks:
            track.predict()

        # ── Step 2 & 3: Associate radar detections ─────────────────────────────
        # Radar supplies [x, y, vx, vy] — position for association, full vector for update.
        # ONLY radar spawns new tracks. Optical false-positive detections (which arrive
        # every frame at ~fp_rate × n_alive) would otherwise create ghost tracks in
        # empty canvas regions — radar has zero false positives so this is safe.
        radar_arr = np.asarray(radar_detections, dtype=float)
        if radar_arr.ndim == 1 and radar_arr.size > 0:
            radar_arr = radar_arr.reshape(1, -1)   # single detection edge-case
        elif radar_arr.size == 0:
            radar_arr = np.empty((0, 4))

        radar_positions = [radar_arr[i, :2] for i in range(len(radar_arr))]
        radar_update_fns = [
            (lambda t, i=i: t.update(radar_arr[i], "radar"))
            for i in range(len(radar_arr))
        ]
        radar_unmatched_idx = self._associate(radar_positions, radar_update_fns)

        # RF confidence gate: when RF data is present (n_sensors=3), use the
        # total number of RF channels monitored (= n_alive, one entry per drone
        # regardless of dropout) as a spawn cap.  Using the channel count rather
        # than only the positive-signal count avoids the systematic under-count
        # caused by the 15% dropout rate (which would cap tracks at ~21 instead
        # of the correct 25 and permanently lose dropped drones).
        rf_arr = np.asarray(rf_detections, dtype=float).ravel()
        rf_count_hint = rf_arr.size if rf_arr.size > 0 else None

        # Unmatched radar detections → spawn new tracks (use reported velocity as seed)
        for i in radar_unmatched_idx:
            if rf_count_hint is not None and len(self._tracks) >= rf_count_hint:
                continue  # RF evidence says we already track enough drones
            det = radar_arr[i]
            self._spawn(x=det[0], y=det[1], vx=det[2], vy=det[3])

        # ── Step 2 & 3: Associate optical detections ───────────────────────────
        # Optical updates existing tracks only — it never spawns new ones.
        # Unmatched optical detections (including false positives) are silently dropped.
        optical_centres = []
        for det in optical_detections:
            bbox = det["bbox"]
            cx = bbox[0] + bbox[2] / 2.0
            cy = bbox[1] + bbox[3] / 2.0
            optical_centres.append(np.array([cx, cy]))

        optical_update_fns = [
            (lambda t, pos=optical_centres[i]: t.update(pos, "optical"))
            for i in range(len(optical_centres))
        ]
        self._associate(optical_centres, optical_update_fns)   # unmatched index ignored

        # ── Step 4: Prune dead tracks ──────────────────────────────────────────
        self._tracks = [t for t in self._tracks if not t.is_dead()]

        # ── Return confirmed tracks only ───────────────────────────────────────
        # Tracks younger than MIN_AGE are still in self._tracks (kept alive so
        # they can accumulate hits) but are hidden from the caller. This prevents
        # optical false-positive ghosts from appearing in the output — they
        # rarely survive long enough to be confirmed.
        self._last_confirmed = [
            {
                "track_id":      t.track_id,
                "position":      t.get_position().tolist(),
                "velocity":      t.get_state()[2:].tolist(),
                "age":           t.age,
                "missed_frames": t.missed_frames,
            }
            for t in self._tracks
            if t.age >= MIN_AGE
        ]
        return self._last_confirmed

    def confirmed_tracks(self) -> list:
        """Return the confirmed tracks produced by the most recent step()."""
        return self._last_confirmed

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _associate(self, positions: list, update_fns: list) -> list:
        """
        Global optimal assignment via Hungarian algorithm.

        Builds a cost matrix C[i, j] = Euclidean distance between track i's
        predicted position and detection j's position. Cells that exceed
        ASSOC_THRESHOLD are set to 1e9 (forbidden). scipy's
        linear_sum_assignment finds the globally minimum-cost assignment;
        pairs whose cost remains <= ASSOC_THRESHOLD are accepted.

        Args:
            positions  : list of (2,) arrays — detection positions
            update_fns : list of callables (track) → None, one per detection

        Returns:
            List of int indices into positions/update_fns that were not matched.
        """
        if not positions or not self._tracks:
            return list(range(len(positions)))

        n_tracks = len(self._tracks)
        n_dets   = len(positions)

        # Build gated cost matrix: rows = tracks, cols = detections
        C = np.full((n_tracks, n_dets), 1e9)
        for i, track in enumerate(self._tracks):
            track_pos = track.get_position()
            for j, pos in enumerate(positions):
                d = float(np.linalg.norm(track_pos - pos))
                if d <= ASSOC_THRESHOLD:
                    C[i, j] = d

        row_idx, col_idx = linear_sum_assignment(C)

        matched_det_ids = set()
        for r, c in zip(row_idx, col_idx):
            if C[r, c] <= ASSOC_THRESHOLD:
                update_fns[c](self._tracks[r])
                matched_det_ids.add(c)

        return [j for j in range(n_dets) if j not in matched_det_ids]

    def _spawn(self, x: float, y: float, vx: float = 0.0, vy: float = 0.0):
        """Initialise a new track at (x, y) with optional velocity seed."""
        filter_cls = DroneEKF if self.use_ekf else DroneKalmanFilter
        track = Track(
            track_id=self._next_id, x=x, y=y, vx=vx, vy=vy, dt=self.dt,
            filter_cls=filter_cls,
        )
        self._tracks.append(track)
        self._next_id += 1


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Smoke test: one target moving right at 1 m/s.
    # dt=1.0 s so positions shift by exactly +1.0 in x each step — easy to verify.
    # We feed a fresh radar detection at the true position each step so the track
    # should stay locked (missed_frames always resets to 0).

    tm = TrackManager(dt=1.0)

    print("TrackManager smoke test — single target moving right at 1 m/s")
    print(f"{'Step':>5}  {'Tracks':>7}  {'x':>8}  {'y':>8}  {'missed':>8}")
    print("-" * 46)

    states = []
    for step in range(10):
        # True position advances by 1.0 each step; radar reports it with no noise.
        det_x = 10.0 + step
        radar = np.array([[det_x, 20.0, 1.0, 0.0]])

        states = tm.step(
            radar_detections=radar,
            optical_detections=[],
            rf_detections=np.empty((0,)),
        )

        for s in states:
            pos = s["position"]
            print(
                f"{step:>5}  {tm.n_tracks:>7}  "
                f"{pos[0]:>8.3f}  {pos[1]:>8.3f}  {s['missed_frames']:>8}"
            )

    # Assertions
    assert tm.n_tracks == 1, f"Expected 1 track, got {tm.n_tracks}"
    assert states[0]["missed_frames"] == 0, "Track should have 0 missed frames"

    final_x = states[0]["position"][0]
    assert abs(final_x - 19.0) < 1.5, f"Expected x ≈ 19.0, got {final_x:.3f}"

    print(f"\nAll assertions passed — 1 track alive, x ≈ {final_x:.3f} after 10 steps.")
