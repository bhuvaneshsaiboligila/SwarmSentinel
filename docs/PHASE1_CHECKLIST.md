# Phase 1 Checklist — Swarm Simulator (Weeks 1–6)

> Updated to reflect actual completion state post-audit-v2.

## Week 1 — Project Setup + Basic Drone
- [x] Create GitHub repository (SwarmSentinel)
- [x] Set up project structure; install numpy, matplotlib
- [x] Drone class: position, velocity, bounds (tuple), alive, kill()
- [x] First commit pushed

## Week 2 — Swarm Class
- [x] Swarm class: `_spawn_drones`, `step()`, behavior dispatch
- [x] Random behavior working; 25-drone simulation runs cleanly

## Week 3 — Flocking Behavior
- [x] `_flock_behavior()`: separation, alignment, cohesion (Reynolds Boids)
- [x] `boundary_forces()` in behaviors.py — per-axis wall repulsion

## Week 4 — Attack Pattern
- [x] `_attack_behavior()`: drones converge on target point
- [x] Empty-alive guard in `_attack_behavior()` (no crash when all dead)
- [x] Attack target marker (red X) visible on plot in run_simulation.py

## Week 5 — Sensor Simulations
- [x] sensors/radar.py — Gaussian noise (σ=1.0 pos, σ=0.3 vel) + 5% miss dropout
- [x] sensors/optical.py — bounding box detections, fp_rate=0.1, bbox clipped to canvas
- [x] sensors/rf.py — signal strength model (max_signal/dist²) + 15% dropout

## Week 6 — Polish + Milestone
- [x] run_simulation.py CLI: `--behavior`, `--mode` alias, `--frames`, `--save`, `--no-display`
- [x] Rectangular bounds: `Drone.bounds` is `(w, h)` tuple; per-axis wrapping and wall repulsion
- [x] Headless mode (`--no-display`): plain for-loop, no FuncAnimation warning
- [x] assets/phase1_demo.gif recorded (304 KB, 100 frames, 15 fps, flock behavior)
- [x] README.md updated with GIF, running instructions, sensor descriptions
- [x] git tag v0.1 applied

---

## Phase 1 Definition of Done

A recruiter should be able to:
1. Clone the repo
2. Run `pip install numpy matplotlib`
3. Run `python simulator/run_simulation.py`
4. Watch a swarm of drones move in real time

✅ Phase 1 is complete.
