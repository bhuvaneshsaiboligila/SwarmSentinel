import numpy as np
from simulator.swarm import Swarm


class RadarSensor:
    """
    Simulates a radar return for every alive drone.

    Returns an (n_detected, 4) array of [x, y, vx, vy] measurements,
    where n_detected <= n_alive after probabilistic miss dropout.
    """

    def __init__(
        self,
        swarm: Swarm,
        pos_sigma: float = 1.0,
        vel_sigma: float = 0.3,
        miss_prob: float = 0.05,
    ):
        self.swarm = swarm
        self.pos_sigma = pos_sigma
        self.vel_sigma = vel_sigma
        self.miss_prob = miss_prob

    def sense(self) -> np.ndarray:
        alive = [d for d in self.swarm.drones if d.alive]
        if not alive:
            return np.empty((0, 4))

        positions = np.array([d.position for d in alive])    # (n, 2)
        velocities = np.array([d.velocity for d in alive])   # (n, 2)

        positions  = positions  + np.random.normal(0.0, self.pos_sigma, positions.shape)
        velocities = velocities + np.random.normal(0.0, self.vel_sigma, velocities.shape)

        measurements = np.hstack([positions, velocities])    # (n, 4)

        keep = np.random.uniform(size=len(alive)) > self.miss_prob
        return measurements[keep]
