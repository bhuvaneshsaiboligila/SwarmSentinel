import numpy as np


def random_forces(positions: np.ndarray, velocities: np.ndarray, n: int) -> np.ndarray:
    """Return (n, 2) array of small random acceleration perturbations."""
    return np.random.normal(0, 0.5, size=(n, 2))


def flock_forces(
    positions: np.ndarray,
    velocities: np.ndarray,
    n: int,
    sep_radius: float,
    align_radius: float,
    cohesion_radius: float,
    w_sep: float = 1.8,
    w_align: float = 1.2,
    w_cohesion: float = 1.0,
    max_force: float = 0.25,
) -> np.ndarray:
    """
    Return (n, 2) combined Boids forces: separation + alignment + cohesion.

    diff[i, j] = positions[j] - positions[i]  →  (n, n, 2)
    Combined force magnitude is clamped to max_force.
    """
    diff = positions[np.newaxis, :, :] - positions[:, np.newaxis, :]
    dists = np.linalg.norm(diff, axis=2)   # (n, n)
    np.fill_diagonal(dists, np.inf)        # exclude self

    # ── Separation ────────────────────────────────────────────────────────────
    sep_mask = dists < sep_radius
    with np.errstate(invalid="ignore", divide="ignore"):
        sep_unit = -diff / dists[:, :, np.newaxis]   # unit vectors toward self
    sep_unit = np.where(sep_mask[:, :, np.newaxis], sep_unit, 0.0)
    f_sep = sep_unit.sum(axis=1) / sep_mask.sum(axis=1, keepdims=True).clip(min=1)

    # ── Alignment ─────────────────────────────────────────────────────────────
    align_mask = dists < align_radius
    neighbor_vel = np.where(align_mask[:, :, np.newaxis], velocities[np.newaxis, :, :], 0.0)
    f_align = (
        neighbor_vel.sum(axis=1) / align_mask.sum(axis=1, keepdims=True).clip(min=1)
        - velocities
    )

    # ── Cohesion ──────────────────────────────────────────────────────────────
    cohesion_mask = dists < cohesion_radius
    # fall back to own position when drone has no neighbors → zero cohesion force
    neighbor_pos = np.where(
        cohesion_mask[:, :, np.newaxis],
        positions[np.newaxis, :, :],
        positions[:, np.newaxis, :],
    )
    center_of_mass = neighbor_pos.sum(axis=1) / cohesion_mask.sum(axis=1, keepdims=True).clip(min=1)
    f_cohesion = center_of_mass - positions

    combined = w_sep * f_sep + w_align * f_align + w_cohesion * f_cohesion

    # Clamp combined force magnitude so no single frame gets an extreme kick
    magnitudes = np.linalg.norm(combined, axis=1, keepdims=True).clip(min=1e-6)
    combined = np.where(magnitudes > max_force,
                        combined * (max_force / magnitudes),
                        combined)
    return combined


def boundary_forces(
    positions: np.ndarray,
    bounds: float,
    margin: float = 20.0,
) -> np.ndarray:
    """
    Return (n, 2) wall-repulsion forces for drones within margin of any boundary.

    Force ramps linearly from 0 at margin distance to 1 at the wall itself,
    directed away from the nearest wall on each axis independently.
    """
    forces = np.zeros_like(positions)
    for axis in range(2):
        dist_lo = positions[:, axis]           # distance from low wall (x=0 or y=0)
        dist_hi = bounds - positions[:, axis]  # distance from high wall
        near_lo = dist_lo < margin
        near_hi = dist_hi < margin
        forces[near_lo, axis] += (margin - dist_lo[near_lo]) / margin
        forces[near_hi, axis] -= (margin - dist_hi[near_hi]) / margin
    return forces


def attack_forces(
    positions: np.ndarray,
    target: np.ndarray,
    n: int,
    max_force: float,
) -> np.ndarray:
    """Return (n, 2) forces steering all drones toward target, capped at max_force."""
    to_target = target - positions                                       # (n, 2)
    dists = np.linalg.norm(to_target, axis=1, keepdims=True).clip(min=1e-6)
    return to_target / dists * np.minimum(dists, max_force)
