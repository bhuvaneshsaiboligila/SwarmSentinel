# CLAUDE.md — SwarmSentinel Project Memory

Read this at the start of every session. Rewrite the relevant sections after every significant change.

---

## Hardware
- OS: Ubuntu 22.04.5 LTS, X11, 64-bit
- CPU: AMD Ryzen 5 3550H (8 threads)
- RAM: 8 GB
- GPU: NVIDIA GTX 1650 Mobile / AMD Radeon Vega Mobile (shared)
- Constraint: animations crash at >25 drones, >15 fps, >2 subplots

## Safe Defaults (never exceed without explicit permission)
- max_drones: 25
- max_fps: 15
- max_subplots: 2
- sensor_update_every_n_frames: 2
- gif_max_frames: 200
- animation_interval_ms: 67

## Python Environment
- Use `/home/bhuvanesh/miniconda3/bin/python` — this is the only interpreter with numpy + matplotlib installed.
- The project venv (`.venv/`) has no packages installed; do not use it.
- Run from project root (`/home/bhuvanesh/projects/SwarmSentinel`) so package imports resolve correctly.

## Current Phase
Phase 1 — COMPLETE | Phase 2 — IN PROGRESS (Weeks 7–12: Sensor Fusion)

---

## Tuned Parameters (behaviors.py / drone.py)

| Parameter | Value | Notes |
|-----------|-------|-------|
| sep_radius | 12.0 | Covers 4% of pairs — prevents overcrowding without scatter |
| align_radius | 45.0 | Covers 41% of pairs — broad heading coherence |
| cohesion_radius | 55.0 | Covers 57% of pairs — satisfies >50% target for cluster formation |
| w_sep | 1.8 | Strong enough to maintain spacing between nearby drones |
| w_align | 1.2 | Slightly above cohesion — heading coherence prioritised |
| w_cohesion | 1.0 | Standard weight; radius scaling handles cluster density |
| max_force (flock) | 0.25 | Per-frame force cap on combined Boids result |
| boundary margin | 20.0 | Repulsion zone wider to compensate for larger canvas coverage |
| max_speed (Drone) | 2.5 | Slightly faster than 2.0; groups still visually trackable |

Boundary repulsion: linear ramp, magnitude 0→1 over the margin zone, per-axis wall-normal direction. Applied additively on top of Boids forces in `_flock_behavior`.

**Diagnostic numbers at 50-step warmup (25 drones, 100×100 canvas):**
- Average inter-drone distance: 50.58 units
- Pairs within sep_radius (12.0):  12 / 300  (4%)
- Pairs within align_radius (45.0): 124 / 300  (41%)
- Pairs within cohesion_radius (55.0): 172 / 300  (57%) ✓ target >50% met
- Centroid: [55.7, 53.5] — near centre
- Spread (std): [25.8, 28.5] — well distributed

---

## What Is Built

### simulator/__init__.py
Empty package initializer. No content needed.

### simulator/drone.py
`Drone` class. Complete.
- `__init__(drone_id, position, velocity, drone_type="scout", max_speed=2.5, bounds=100.0)`
- `update(dt)`: advances position, wraps via `% self.bounds`, clamps speed to `max_speed`
- `apply_force(force, dt)`: integrates acceleration into velocity
- `kill()`: sets `alive=False`, zeros velocity
- `distance_to(other)`: Euclidean distance

### simulator/swarm.py
`Swarm` class. Complete.
- `__init__(n_drones, spawn_area, behavior="random", target=None)`
- `step(dt)`: dispatches to `_random_behavior`, `_flock_behavior`, or `_attack_behavior`; raises `ValueError` on unknown behavior
- Delegates all force math to `simulator.behaviors` (no inline math)
- `get_positions()` → (N, 2), `get_velocities()` → (N, 2), `n_alive` property
- **Known issue**: `_spawn_drones` does not pass `bounds=` to `Drone`; drones always wrap at 100.0 regardless of `spawn_area`. `spawn_area` must therefore always be `(100, 100)` or smaller, otherwise drones immediately wrap to [0, 100) and the visualization axis will be wrong.

### simulator/behaviors.py
Four standalone pure-NumPy functions. Complete.
- `random_forces(positions, velocities, n)` → (n, 2): Gaussian noise σ=0.5
- `flock_forces(positions, velocities, n, sep_radius, align_radius, cohesion_radius, w_sep=1.5, w_align=1.0, w_cohesion=1.0)` → (n, 2): vectorised Boids — builds (N,N,2) diff tensor, reuses it for all three rules
- `boundary_forces(positions, bounds, margin=20.0)` → (n, 2): linear wall-repulsion force away from each arena edge
- `attack_forces(positions, target, n, max_force)` → (n, 2): unit-vector steering capped at max_force

### simulator/run_simulation.py
Single-subplot live animation + CLI entry point. Hardware-safe.
- `--behavior random|flock|attack` (default: flock)
- `--save PATH`: writes GIF via `PillowWriter` instead of opening a window
- Attack target hardcoded to `[50.0, 50.0]` (centre of 100×100 world)
- Safe constants: `N_DRONES=25`, `DT=1/15`, `MAX_STEPS=200`, `INTERVAL=67`
- Single subplot showing ground truth positions only
- Simulation time tracked via `t_sim` accumulator (not `frame * DT`) so time display is correct when `--save` is used

### sensors/__init__.py
Empty package initializer.

### sensors/radar.py
`RadarSensor`. Complete.
- `sense()` → ndarray (n_detected, 4): [x, y, vx, vy] with Gaussian noise (pos σ=1.0, vel σ=0.3) and 5% miss dropout

### sensors/optical.py
`OpticalSensor`. Complete.
- `sense()` → list of `{drone_id, bbox: [x,y,w,h], confidence}` dicts
- 10% miss dropout; false positives via `Binomial(n_alive, fp_rate=0.1)` placed uniformly over `area`
- `drone_id=None` marks false positives
- Default `area=(500, 500)` in the constructor — **always pass `area=AREA` explicitly** so FPs are placed within the visible canvas.

### sensors/rf.py
`RFSensor`. Complete.
- `sense()` → ndarray (n_alive,): signal = `max_signal / dist²` + Gaussian noise (σ=0.5); 15% dropout sets signal to 0.0 exactly
- Signals can be negative for distant drones (noise floor — intentional, not a bug)

---

## What Is Broken / Known Issues

1. **Swarm does not propagate bounds to Drone.** `spawn_area` must always equal `(100, 100)` or any area ≤ 100 on both axes. Any larger area causes silent wrap-to-[0,100) at the first `update()` call.
2. **OpticalSensor default area mismatch.** Constructor defaults to `area=(500,500)` but the simulation uses `(100,100)`. Always pass `area=AREA` explicitly at construction.
3. **`python fusion/file.py` requires sys.path fix.** Running any fusion script with `python fusion/file.py` puts `fusion/` (not the project root) in `sys.path[0]`, breaking `from fusion.* import`. Fixed in `track_manager.py` and `run_fusion.py` via `sys.path.insert(0, project_root)` guarded by `if __name__ == "__main__"`. Same applies to `simulator/run_simulation.py`.
4. **Track count inflated to 30+ tracks for 25 drones — fixed.** Root cause: optical false positives (~2–3 per frame, fp_rate=0.1, 25 drones) landed in empty canvas regions and spawned ghost tracks that aged past the min_age gate by absorbing subsequent FPs through the wide association gate. Fix: optical detections now only UPDATE existing tracks, never spawn new ones. Only radar spawns (radar has 0% FP rate). Result: steady-state output is exactly 25 tracks.
5. **`--save` flag requires an explicit PATH.** `python fusion/run_fusion.py --save` without a path raises argparse error. Must pass full path: `python fusion/run_fusion.py --save assets/phase2_fusion_demo.gif`.
6. **Ignored-vs-tracked asset rules are inconsistent.** `.gitignore` excludes new `assets/*.gif` files and also lists `CLAUDE.md`, but `CLAUDE.md` and `assets/phase2_fusion_demo.gif` are already tracked. In practice: `assets/phase1_demo.gif` is still ignored/untracked; tracked files continue to update normally; any brand-new GIF still needs `git add -f`.

---

## Last Tagged Milestone
v0.1 — Phase 1 complete. env_check passes (Python 3.13.11, NumPy 2.4.4, Matplotlib 3.10.9). run_simulation.py constants fixed to safe values (N=25, 15fps, 200 frames, 1 subplot). README updated with GIF embed, running instructions, sensor model descriptions, Phase 1 marked Complete.

## Last Working Demo
`assets/phase2_fusion_demo.gif` — 150 frames, 15 fps, 1200×600 px, 2.1 MB. Two-subplot animation: raw sensor detections (left) vs. fused Kalman tracks (right). 25 drones, flock behavior.

## Next Task
Phase 2 — EKF (`fusion/ekf.py`) and fusion error benchmark (1 vs 2 vs 3 sensors)

---

## Phase 1 Checklist

- [x] simulator/__init__.py
- [x] drone.py with bounds wrapping + max_speed clamp
- [x] swarm.py with real flock + attack behaviors
- [x] behaviors.py standalone vectorised functions
- [x] run_simulation.py safe for hardware (validated at 25 drones, 15 fps, 1 subplot, 200 frames)
- [x] sensors/radar.py
- [x] sensors/optical.py
- [x] sensors/rf.py
- [x] env_check script passes cleanly (Python 3.13.11 / NumPy 2.4.4 / Matplotlib 3.10.9)
- [x] assets/phase1_demo.gif recorded (304 KB, 100 frames, 15 fps, flock behavior)
- [x] git tag v0.1

---

## Phase 2 Progress (Weeks 7–12) — IN PROGRESS

- [x] Basic Kalman filter (`DroneKalmanFilter`) implemented — `fusion/kalman_filter.py`
  - State vector: `[x, y, vx, vy]` (4D)
  - Measurement modes: radar (4D: x, y, vx, vy) and optical (2D: x, y only)
  - Motion model: constant velocity with discrete white-noise acceleration Q
  - R matrices matched to sensor noise σ values in `sensors/radar.py` / `sensors/optical.py`
  - Optical update uses raw Kalman equations (not filterpy's kf.update) — filterpy validates dim_z at call time and rejects shape (2,) when dim_z=4
- [x] Multi-target track manager — `fusion/track_manager.py`
  - `ASSOC_THRESHOLD = 15.0` m (Euclidean, greedy nearest-neighbour) — tuning param
  - `MAX_MISSED = 8` frames before a track is pruned — tuning param
  - `MIN_AGE = 3` frames before a track appears in output — kills transient ghosts — tuning param
  - **Only radar spawns new tracks; optical only updates existing tracks.**
    Optical FPs (≈2–3 per frame) in empty canvas regions were creating ghost tracks.
    Radar has 0% false-positive rate so radar-only spawning eliminates this entirely.
  - Result: steady-state output = exactly 25 tracks for 25 drones (mean=25.0, min=25, max=25)
  - RF accepted in signature but unused — signal strength carries no position info
- [x] Fusion pipeline (sensors → TrackManager) — `fusion/fusion_pipeline.py`
  - Derives `area` from `swarm.drones[0].bounds` to fix OpticalSensor FP placement
- [x] Side-by-side visualisation — `fusion/run_fusion.py`
  - 2 subplots, 25 drones, 15 fps, 150 frames (all within hardware limits)
  - RF shown as text count (not dots) — no spatial position output from RFSensor
  - `--save PATH` writes GIF via PillowWriter; `--save` alone requires explicit PATH
  - `run_fusion.py` wires sensors + TrackManager directly (does not use FusionPipeline class)
- [x] EKF stub — `fusion/ekf.py`
  - `DroneEKF` class, identical interface to `DroneKalmanFilter` (drop-in swap)
  - Currently uses linear constant-velocity model — reduces to standard KF
  - TODO: replace with constant-turn model (adds turn-rate state `omega`, trig f(x)) in Phase 2 week 2
  - predict() sets F Jacobian fresh each call so nonlinear f(x) can be swapped in without restructuring
- [x] Benchmark skeleton — `benchmarks/fusion_error.py`
  - `benchmark_sensor_count(n_sensors: int) -> float` exists, returns `0.0` placeholder
  - TODO: implement RMSE between fused tracks and ground truth; compare n_sensors = 1, 2, 3
- [ ] git tag v0.2
