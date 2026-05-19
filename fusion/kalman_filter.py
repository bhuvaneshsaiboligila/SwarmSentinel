"""
fusion/kalman_filter.py — Standard Kalman Filter for single drone tracking.

State vector : [x, y, vx, vy]
Motion model : constant velocity  (x += vx*dt, y += vy*dt each step)
Sensors      : radar [x, y, vx, vy]  or  optical [x, y] only

──────────────────────────────────────────────────────────────────────────────
A note on Q and R before you read the code:

  Q — Process noise covariance
      How uncertain we are about the motion model itself.
      Drones can accelerate, turn, or jink — the constant-velocity model
      can't predict that. Q encodes "how wrong might the model be per step?"
      Large Q  → filter reacts quickly to measurements, noisier estimate.
      Small Q  → filter trusts its own prediction, slow to track manoeuvres.

  R — Measurement noise covariance
      How noisy each sensor is.
      Large R  → sensor is unreliable; filter leans on its own prediction.
      Small R  → sensor is precise; filter snaps hard to measurements.

  The values below are matched to the noise models in sensors/:
      Radar pos σ = 1.0 m   → var = 1.0
      Radar vel σ = 0.3 m/s → var = 0.09
      Optical pos σ = 2.0 m → var = 4.0
──────────────────────────────────────────────────────────────────────────────
"""

import numpy as np
from filterpy.kalman import KalmanFilter


class DroneKalmanFilter:
    """
    Linear Kalman Filter tracking one drone in 2D.

    Uses filterpy's KalmanFilter (dim_x=4, dim_z=4) for radar updates.
    Optical updates are computed with the raw Kalman equations because
    the measurement dimension differs (2 vs 4) and filterpy validates dim_z
    at call time — swapping H/R plus manual equations is cleaner.
    """

    def __init__(self, dt: float = 1 / 15):
        """
        Args:
            dt: timestep in seconds. Should match the simulation's DT (default 1/15 s).
        """
        self.dt = dt

        # filterpy KalmanFilter: 4-dimensional state, 4-dimensional measurement (radar)
        self.kf = KalmanFilter(dim_x=4, dim_z=4)

        # ── State transition matrix F ──────────────────────────────────────────
        # Constant-velocity model: position advances by velocity * dt each step.
        #   x_new  = x  + vx * dt
        #   y_new  = y  + vy * dt
        #   vx_new = vx               (no acceleration assumed)
        #   vy_new = vy
        self.kf.F = np.array([
            [1, 0, dt,  0],
            [0, 1,  0, dt],
            [0, 0,  1,  0],
            [0, 0,  0,  1],
        ], dtype=float)

        # ── Process noise Q ────────────────────────────────────────────────────
        # Discrete white-noise acceleration model (Singer 1970).
        # σ_a = acceleration noise std; here q = σ_a² = 0.1 (mild manoeuvring).
        # The off-diagonal terms couple position and velocity uncertainty
        # correctly over the timestep.
        q = 0.1
        self.kf.Q = np.array([
            [q * dt**4 / 4, 0,             q * dt**3 / 2, 0            ],
            [0,             q * dt**4 / 4, 0,             q * dt**3 / 2],
            [q * dt**3 / 2, 0,             q * dt**2,     0            ],
            [0,             q * dt**3 / 2, 0,             q * dt**2    ],
        ])

        # ── Initial state covariance P ─────────────────────────────────────────
        # Large diagonal = high uncertainty at start.
        # The filter will tighten P as measurements arrive.
        self.kf.P = np.diag([100.0, 100.0, 10.0, 10.0])

        # ── Measurement matrices ───────────────────────────────────────────────
        # Radar sees all four state variables directly.
        self._H_radar = np.eye(4)

        # Optical sees position only; velocity rows are zeroed out.
        self._H_optical = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=float)

        # ── Measurement noise R (per sensor) ──────────────────────────────────
        # Values match the noise parameters in sensors/radar.py and sensors/optical.py.
        self._R_radar = np.diag([1.0, 1.0, 0.09, 0.09])   # pos var=1, vel var=0.09
        self._R_optical = np.diag([4.0, 4.0])              # pos var=4 (σ=2 m)

        # Default H/R to radar so filterpy's internal state is consistent.
        self.kf.H = self._H_radar
        self.kf.R = self._R_radar

    def initialize(self, x: float, y: float, vx: float = 0.0, vy: float = 0.0):
        """Seed the filter with a known or estimated starting state."""
        self.kf.x = np.array([[x], [y], [vx], [vy]], dtype=float)

    def predict(self):
        """Advance the state estimate by one timestep using the motion model."""
        self.kf.predict()

    def radar_update(self, measurement: np.ndarray):
        """
        Fuse a radar measurement into the state estimate.

        Args:
            measurement: shape (4,) — [x, y, vx, vy] from RadarSensor.sense()
        """
        self.kf.H = self._H_radar
        self.kf.R = self._R_radar
        self.kf.update(measurement.reshape(4, 1))

    def optical_update(self, measurement: np.ndarray):
        """
        Fuse an optical measurement (position only) into the state estimate.

        Filterpy validates that z.shape matches dim_z, so we apply the Kalman
        update equations directly here rather than going through kf.update().

        Args:
            measurement: shape (2,) — [x, y] bbox centre from OpticalSensor.sense()
        """
        H = self._H_optical     # (2, 4)
        R = self._R_optical     # (2, 2)
        x = self.kf.x           # (4, 1) current state
        P = self.kf.P           # (4, 4) current covariance

        # Innovation: how far the measurement is from our prediction
        y = measurement.reshape(2, 1) - H @ x               # (2, 1)

        # Innovation covariance: prediction uncertainty projected into sensor space
        S = H @ P @ H.T + R                                  # (2, 2)

        # Kalman gain: blend factor between prediction and measurement
        K = P @ H.T @ np.linalg.inv(S)                      # (4, 2)

        # Updated state: pull prediction toward measurement, weighted by K
        self.kf.x = x + K @ y                               # (4, 1)

        # Updated covariance: Joseph form for numerical stability
        # (I - KH) P (I - KH)ᵀ + K R Kᵀ  stays symmetric and positive-definite
        I_KH = np.eye(4) - K @ H                            # (4, 4)
        self.kf.P = I_KH @ P @ I_KH.T + K @ R @ K.T        # (4, 4)

    def get_state(self) -> np.ndarray:
        """Return current state estimate as a flat array: [x, y, vx, vy]."""
        return self.kf.x.flatten()


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Quick smoke-test: predict-only, no sensor updates.
    # Starting at (10, 20) with velocity (1.0, 0.5) and dt=1.0 s,
    # each step should advance x by ~1.0 and y by ~0.5.

    DT = 1.0
    kf = DroneKalmanFilter(dt=DT)
    kf.initialize(x=10.0, y=20.0, vx=1.0, vy=0.5)

    print("DroneKalmanFilter — predict-only test (dt=1.0 s)")
    print(f"{'Step':>5}  {'x':>8}  {'y':>8}  {'vx':>8}  {'vy':>8}")
    print("-" * 48)

    s = kf.get_state()
    print(f"{'init':>5}  {s[0]:>8.3f}  {s[1]:>8.3f}  {s[2]:>8.3f}  {s[3]:>8.3f}")

    for step in range(1, 6):
        kf.predict()
        s = kf.get_state()
        print(f"{step:>5}  {s[0]:>8.3f}  {s[1]:>8.3f}  {s[2]:>8.3f}  {s[3]:>8.3f}")

    # Verify direction: after 5 steps at vx=1.0, vy=0.5 the drone must have moved right and up
    assert s[0] > 10.0, f"x should have increased from 10.0, got {s[0]:.3f}"
    assert s[1] > 20.0, f"y should have increased from 20.0, got {s[1]:.3f}"
    # Velocity should be unchanged (no forces, no measurements)
    assert abs(s[2] - 1.0) < 0.01, f"vx should stay ~1.0, got {s[2]:.3f}"
    assert abs(s[3] - 0.5) < 0.01, f"vy should stay ~0.5, got {s[3]:.3f}"

    print("\nAll assertions passed — filter moves in the correct direction.")
