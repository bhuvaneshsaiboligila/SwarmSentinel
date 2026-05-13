# drone.py — Single drone entity
# Phase 1: SwarmSentinel
#
# This is your starting point. A single Drone object that knows
# its position, velocity, and how to move each timestep.

import numpy as np


class Drone:
    """
    Represents a single drone in the simulation.

    Attributes:
        drone_id  : unique integer identifier
        position  : np.array([x, y]) in meters
        velocity  : np.array([vx, vy]) in meters/second
        drone_type: 'scout', 'attack', or 'decoy'
    """

    def __init__(
        self,
        drone_id: int,
        position: np.ndarray,
        velocity: np.ndarray,
        drone_type: str = "scout",
        max_speed: float = 2.5,
        bounds: float = 100.0,
    ):
        self.drone_id = drone_id
        self.position = position.astype(float)
        self.velocity = velocity.astype(float)
        self.drone_type = drone_type
        self.alive = True
        self.max_speed = max_speed
        self.bounds = bounds

    def update(self, dt: float):
        """
        Move drone forward by one timestep.

        Args:
            dt: timestep in seconds (e.g. 0.1 for 10Hz update rate)
        """
        self.position += self.velocity * dt
        self.position[:] = self.position % self.bounds
        speed = np.linalg.norm(self.velocity)
        if speed > self.max_speed:
            self.velocity *= self.max_speed / speed

    def apply_force(self, force: np.ndarray, dt: float):
        """
        Apply an acceleration force to the drone.
        Used by swarm behavior algorithms.

        Args:
            force: np.array([fx, fy]) acceleration in m/s^2
            dt   : timestep in seconds
        """
        self.velocity += force * dt

    def kill(self):
        """Mark drone as destroyed and zero its velocity."""
        self.alive = False
        self.velocity[:] = 0.0

    def distance_to(self, other: "Drone") -> float:
        """Returns Euclidean distance to another drone in meters."""
        return float(np.linalg.norm(self.position - other.position))

    def __repr__(self):
        return (
            f"Drone(id={self.drone_id}, type={self.drone_type}, "
            f"pos={self.position.round(1)}, vel={self.velocity.round(1)})"
        )
