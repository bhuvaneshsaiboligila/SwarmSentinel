import numpy as np
from simulator.swarm import Swarm


class RFSensor:
    """
    Simulates RF signal-strength measurements for alive drones.

    Returns an (n_alive,) array of signal-strength values.
    Signal model: max_signal / distance_to_origin²
    Dropped detections are returned as 0.0.
    """

    def __init__(
        self,
        swarm: Swarm,
        max_signal: float = 100.0,
        noise_sigma: float = 0.5,
        dropout_prob: float = 0.15,
    ):
        self.swarm = swarm
        self.max_signal = max_signal
        self.noise_sigma = noise_sigma
        self.dropout_prob = dropout_prob

    def sense(self) -> np.ndarray:
        alive = [d for d in self.swarm.drones if d.alive]
        if not alive:
            return np.empty((0,))

        positions = np.array([d.position for d in alive])              # (n, 2)
        dist_sq   = np.sum(positions ** 2, axis=1).clip(min=1e-6)      # (n,)
        signals   = self.max_signal / dist_sq
        signals  += np.random.normal(0.0, self.noise_sigma, len(alive))

        dropout           = np.random.uniform(size=len(alive)) < self.dropout_prob
        signals[dropout]  = 0.0

        return signals
