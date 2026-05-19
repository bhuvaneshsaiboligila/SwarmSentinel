"""
fusion/ekf.py — Extended Kalman Filter stub for single drone tracking.

Interface is identical to DroneKalmanFilter so it can be swapped in
anywhere DroneKalmanFilter is used.

Current state: linear motion model (constant velocity), same as the
standard KF. This is a valid EKF — just one whose f(x) happens to be
linear, so it reduces to an ordinary KF.

TODO: Replace linear motion model with a constant-turn (CT) model in
      Phase 2 week 2. The CT model adds a turn-rate state omega and uses
      trigonometric f(x) / F_jacobian — that is where the EKF becomes
      meaningfully different from the standard KF.
"""

import numpy as np
from filterpy.kalman import ExtendedKalmanFilter


def _F_jacobian(x, dt):
    """Jacobian of the (linear) state transition function f(x) w.r.t. x."""
    return np.array([
        [1, 0, dt,  0],
        [0, 1,  0, dt],
        [0, 0,  1,  0],
        [0, 0,  0,  1],
    ], dtype=float)


def _f(x, dt):
    """State transition function: constant-velocity step."""
    F = _F_jacobian(x, dt)
    return F @ x


class DroneEKF:
    """
    Extended Kalman Filter tracking one drone in 2D.

    State vector : [x, y, vx, vy]
    Motion model : constant velocity (placeholder — see module TODO above)
    Sensors      : radar [x, y, vx, vy]  or  optical [x, y]

    Identical public interface to DroneKalmanFilter so either class can
    be passed to TrackManager or FusionPipeline without changes.
    """

    def __init__(self, dt: float = 1 / 15):
        self.dt = dt

        # filterpy EKF: 4-D state, 4-D default measurement (radar)
        self.ekf = ExtendedKalmanFilter(dim_x=4, dim_z=4)

        # ── Process noise Q ────────────────────────────────────────────────────
        q = 0.1
        self.ekf.Q = np.array([
            [q * dt**4 / 4, 0,             q * dt**3 / 2, 0            ],
            [0,             q * dt**4 / 4, 0,             q * dt**3 / 2],
            [q * dt**3 / 2, 0,             q * dt**2,     0            ],
            [0,             q * dt**3 / 2, 0,             q * dt**2    ],
        ])

        # ── Initial state covariance P ─────────────────────────────────────────
        self.ekf.P = np.diag([100.0, 100.0, 10.0, 10.0])

        # ── Measurement matrices ───────────────────────────────────────────────
        self._H_radar   = np.eye(4)
        self._H_optical = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)

        # ── Measurement noise R ────────────────────────────────────────────────
        self._R_radar   = np.diag([1.0, 1.0, 0.09, 0.09])
        self._R_optical = np.diag([4.0, 4.0])

    def initialize(self, x: float, y: float, vx: float = 0.0, vy: float = 0.0):
        """Seed the filter with a known or estimated starting state."""
        self.ekf.x = np.array([[x], [y], [vx], [vy]], dtype=float)

    def predict(self):
        """
        Propagate state forward using the nonlinear motion function f(x).

        filterpy's EKF.predict() calls self.ekf.F as the Jacobian; we set
        it fresh each call so that a future nonlinear f(x) can update F here.
        """
        self.ekf.F = _F_jacobian(self.ekf.x, self.dt)
        self.ekf.x = _f(self.ekf.x, self.dt)
        self.ekf.P = self.ekf.F @ self.ekf.P @ self.ekf.F.T + self.ekf.Q

    def radar_update(self, measurement: np.ndarray):
        """Fuse a radar measurement [x, y, vx, vy] into the state estimate."""
        H = self._H_radar
        R = self._R_radar
        z = measurement.reshape(4, 1)
        self._ekf_update(H, R, z)

    def optical_update(self, measurement: np.ndarray):
        """Fuse an optical measurement [x, y] (position only) into the state estimate."""
        H = self._H_optical
        R = self._R_optical
        z = measurement.reshape(2, 1)
        self._ekf_update(H, R, z)

    def _ekf_update(self, H: np.ndarray, R: np.ndarray, z: np.ndarray):
        """
        Standard linear Kalman update step (valid because H is linear here).
        When f(x) becomes nonlinear in week 2, only predict() changes;
        this update step stays the same if h(x) = H @ x remains linear.
        Uses Joseph form for numerical stability.
        """
        x, P = self.ekf.x, self.ekf.P
        y = z - H @ x
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)
        self.ekf.x = x + K @ y
        I_KH = np.eye(4) - K @ H
        self.ekf.P = I_KH @ P @ I_KH.T + K @ R @ K.T

    def get_state(self) -> np.ndarray:
        """Return current state estimate as a flat array: [x, y, vx, vy]."""
        return self.ekf.x.flatten()
