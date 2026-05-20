"""
fusion/ekf.py — Extended Kalman Filter for single drone tracking.

State vector : [x, y, vx, vy, omega]   (omega = turn rate, rad/s)
Motion model : Constant Turn Rate (CTR)
    x_new     = x  + vx*dt
    y_new     = y  + vy*dt
    vx_new    = vx*cos(omega*dt) - vy*sin(omega*dt)
    vy_new    = vx*sin(omega*dt) + vy*cos(omega*dt)
    omega_new = omega

Public interface is identical to DroneKalmanFilter (predict / radar_update /
optical_update / get_state) so either class can be dropped into TrackManager
or FusionPipeline without code changes.
"""

import sys
import pathlib

if __name__ == "__main__":
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from filterpy.kalman import ExtendedKalmanFilter


def _f(x, dt):
    """CTR state transition — returns new 5-D state column vector."""
    px, py, vx, vy, om = x.flatten()
    c = np.cos(om * dt)
    s = np.sin(om * dt)
    return np.array([
        [px + vx * dt],
        [py + vy * dt],
        [vx * c - vy * s],
        [vx * s + vy * c],
        [om],
    ], dtype=float)


def _F_jacobian(x, dt):
    """Analytical Jacobian of f w.r.t. x — 5×5 matrix."""
    _, _, vx, vy, om = x.flatten()
    c = np.cos(om * dt)
    s = np.sin(om * dt)
    return np.array([
        [1, 0, dt,  0,  0],
        [0, 1,  0, dt,  0],
        [0, 0,  c, -s,  dt * (-vx * s - vy * c)],
        [0, 0,  s,  c,  dt * ( vx * c - vy * s)],
        [0, 0,  0,  0,  1],
    ], dtype=float)


class DroneEKF:
    """
    Extended Kalman Filter tracking one drone in 2D.

    State vector : [x, y, vx, vy, omega]
    Motion model : Constant Turn Rate (CTR) with analytic Jacobian
    Sensors      : radar [x, y, vx, vy]  or  optical [x, y]
    """

    def __init__(self, dt: float = 1 / 15):
        self.dt = dt

        # filterpy EKF: 5-D state, 4-D default measurement slot (radar)
        self.ekf = ExtendedKalmanFilter(dim_x=5, dim_z=4)

        # ── Process noise Q (conservative) ────────────────────────────────────
        self.ekf.Q = np.eye(5) * 0.1

        # ── Initial state covariance P ─────────────────────────────────────────
        # omega starts at 0 with low variance (0.01) — drones rarely spin fast.
        self.ekf.P = np.diag([100.0, 100.0, 10.0, 10.0, 0.01])

        # ── Measurement matrices (linear h(x) = H @ x) ────────────────────────
        self._H_radar = np.array([
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
        ], dtype=float)   # 4×5

        self._H_optical = np.array([
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
        ], dtype=float)   # 2×5

        # ── Measurement noise R (matched to sensor σ in sensors/*.py) ──────────
        self._R_radar   = np.diag([1.0, 1.0, 0.09, 0.09])   # pos σ=1, vel σ=0.3
        self._R_optical = np.diag([4.0, 4.0])                # pos σ=2

    def initialize(self, x: float, y: float, vx: float = 0.0, vy: float = 0.0):
        """Seed the filter. omega is initialised to 0.0 (straight-line prior)."""
        self.ekf.x = np.array([[x], [y], [vx], [vy], [0.0]], dtype=float)

    def predict(self, dt: float = None):
        """
        Propagate state forward using the nonlinear CTR motion model.
        dt is optional so TrackManager can call predict() without arguments.
        """
        if dt is None:
            dt = self.dt
        F = _F_jacobian(self.ekf.x, dt)
        self.ekf.x = _f(self.ekf.x, dt)
        self.ekf.P = F @ self.ekf.P @ F.T + self.ekf.Q

    def radar_update(self, measurement: np.ndarray):
        """Fuse a radar measurement [x, y, vx, vy] into the state estimate."""
        self._update(self._H_radar, self._R_radar, measurement.reshape(-1, 1))

    def optical_update(self, measurement: np.ndarray):
        """Fuse an optical measurement [x, y] (position only) into the state estimate."""
        self._update(self._H_optical, self._R_optical, measurement.reshape(-1, 1))

    def _update(self, H: np.ndarray, R: np.ndarray, z: np.ndarray):
        """
        Linear Kalman update in Joseph form for numerical stability.
        Valid because h(x) = H @ x is linear even though f(x) is not.
        """
        x, P = self.ekf.x, self.ekf.P
        y    = z - H @ x
        S    = H @ P @ H.T + R
        K    = P @ H.T @ np.linalg.inv(S)
        self.ekf.x = x + K @ y
        I_KH = np.eye(5) - K @ H
        self.ekf.P = I_KH @ P @ I_KH.T + K @ R @ K.T

    def get_state(self) -> np.ndarray:
        """Return current state estimate as a flat array: [x, y, vx, vy, omega]."""
        return self.ekf.x.flatten()


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Smoke test: drone flying a circular arc (constant omega = 0.3 rad/s).
    # Feed noisy radar measurements each step and print position residuals.

    DT    = 1 / 15
    OMEGA = 0.3   # rad/s

    ekf = DroneEKF(dt=DT)
    ekf.initialize(x=10.0, y=10.0, vx=2.0, vy=0.0)

    rng      = np.random.default_rng(42)
    true_pos = np.array([10.0, 10.0])
    true_vel = np.array([2.0, 0.0])

    print("DroneEKF CTR smoke test — circular arc, 10 steps")
    print(f"{'step':>5}  {'true_x':>8}  {'true_y':>8}  "
          f"{'est_x':>8}  {'est_y':>8}  {'res_x':>7}  {'res_y':>7}  {'omega':>7}")
    print("-" * 74)

    for step in range(10):
        # Advance ground truth one CTR step
        c = np.cos(OMEGA * DT)
        s = np.sin(OMEGA * DT)
        true_pos = true_pos + true_vel * DT
        true_vel = np.array([true_vel[0]*c - true_vel[1]*s,
                              true_vel[0]*s + true_vel[1]*c])

        # Noisy radar measurement
        z_noise = rng.normal(0.0, [1.0, 1.0, 0.3, 0.3])
        z = np.array([true_pos[0], true_pos[1],
                      true_vel[0], true_vel[1]]) + z_noise

        ekf.predict(DT)
        ekf.radar_update(z)

        st   = ekf.get_state()
        rx   = true_pos[0] - st[0]
        ry   = true_pos[1] - st[1]

        print(f"{step:>5}  {true_pos[0]:>8.3f}  {true_pos[1]:>8.3f}  "
              f"{st[0]:>8.3f}  {st[1]:>8.3f}  {rx:>7.3f}  {ry:>7.3f}  {st[4]:>7.4f}")

    final_rmse = float(np.sqrt((true_pos[0] - st[0])**2 + (true_pos[1] - st[1])**2))
    assert final_rmse < 5.0, f"Position residual too large: {final_rmse:.3f} m"
    print(f"\nFinal position RMSE {final_rmse:.3f} m — EKF tracking OK.")
