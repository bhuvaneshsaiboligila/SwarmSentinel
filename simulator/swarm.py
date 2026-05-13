# swarm.py — Swarm controller
# Phase 1: SwarmSentinel
#
# The Swarm class manages a group of Drone objects.
# It runs the behavior algorithm each timestep and updates all drones.

import numpy as np
from simulator.drone import Drone
from simulator.behaviors import random_forces, flock_forces, attack_forces, boundary_forces


class Swarm:
    """
    Manages a collection of drones and applies swarm behavior.

    Args:
        n_drones  : number of drones in the swarm
        spawn_area: (width, height) in meters — drones spawn randomly inside this
        behavior  : 'random', 'flock', or 'attack'
        target    : np.array([x, y]) — used only if behavior == 'attack'
    """

    def __init__(self, n_drones: int, spawn_area: tuple, behavior: str = "random", target: np.ndarray = None):
        self.behavior = behavior
        self.target = target
        self.drones = self._spawn_drones(n_drones, spawn_area)

    def _spawn_drones(self, n: int, area: tuple) -> list:
        drones = []
        for i in range(n):
            pos = np.random.uniform([0, 0], [area[0], area[1]])
            vel = np.random.uniform(-2.0, 2.0, size=2)  # random initial velocity
            drones.append(Drone(drone_id=i, position=pos, velocity=vel))
        return drones

    def step(self, dt: float):
        """Advance all drones by one timestep using the current behavior."""
        if self.behavior == "random":
            self._random_behavior(dt)
        elif self.behavior == "flock":
            self._flock_behavior(dt)
        elif self.behavior == "attack":
            self._attack_behavior(dt)
        else:
            raise ValueError(f"Unknown behavior: {self.behavior!r}. Expected 'random', 'flock', or 'attack'.")

        for drone in self.drones:
            drone.update(dt)

    def _random_behavior(self, dt: float):
        """Add small random perturbations — simulates disorganized movement."""
        positions = np.array([d.position for d in self.drones])
        velocities = np.array([d.velocity for d in self.drones])
        forces = random_forces(positions, velocities, len(self.drones))
        for drone, force in zip(self.drones, forces):
            drone.apply_force(force, dt)

    def _flock_behavior(
        self,
        dt: float,
        sep_radius: float = 12.0,
        align_radius: float = 45.0,
        cohesion_radius: float = 55.0,
    ):
        """Classic Reynolds Boids: separation, alignment, cohesion + wall repulsion."""
        positions = np.array([d.position for d in self.drones])
        velocities = np.array([d.velocity for d in self.drones])
        bounds = self.drones[0].bounds
        forces = (
            flock_forces(positions, velocities, len(self.drones),
                         sep_radius, align_radius, cohesion_radius)
            + boundary_forces(positions, bounds)
        )
        for drone, force in zip(self.drones, forces):
            drone.apply_force(force, dt)

    def _attack_behavior(self, dt: float, max_force: float = 0.5):
        """All drones accelerate toward self.target, capped at max_force."""
        if self.target is None:
            self._flock_behavior(dt)
            return
        positions = np.array([d.position for d in self.drones])
        forces = attack_forces(positions, self.target, len(self.drones), max_force)
        for drone, force in zip(self.drones, forces):
            drone.apply_force(force, dt)

    def get_positions(self) -> np.ndarray:
        """Returns (N, 2) array of all drone positions."""
        return np.array([d.position for d in self.drones])

    def get_velocities(self) -> np.ndarray:
        """Returns (N, 2) array of all drone velocities."""
        return np.array([d.velocity for d in self.drones])

    @property
    def n_alive(self) -> int:
        return sum(1 for d in self.drones if d.alive)
