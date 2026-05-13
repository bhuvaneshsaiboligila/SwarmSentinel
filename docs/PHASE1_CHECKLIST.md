# Phase 1 Checklist — Swarm Simulator (Weeks 1–6)

Mark each item with [x] when done. Commit after each checkbox.

## Week 1 — Project Setup + Basic Drone
- [ ] Create GitHub repository (name: SwarmSentinel)
- [ ] Clone repo locally, set up project structure
- [ ] Install requirements: numpy, matplotlib
- [ ] Read and understand drone.py — run it in a Python shell
- [ ] Write a script that creates 5 drones and prints their positions
- [ ] Push first commit: "feat: project scaffold and Drone class"

## Week 2 — Swarm Class
- [ ] Read swarm.py fully
- [ ] Create 20 drones using Swarm class
- [ ] Run run_simulation.py and see drones moving randomly
- [ ] Understand what `step()` does
- [ ] Push commit: "feat: Swarm class with random behavior"

## Week 3 — Flocking Behavior
- [ ] Research Reynolds flocking rules (separation, alignment, cohesion)
- [ ] Implement _flock_behavior() in swarm.py
- [ ] Run simulation: do drones move together? Tune the weights
- [ ] Compare random vs. flock visually (run both)
- [ ] Push commit: "feat: Reynolds flocking algorithm"

## Week 4 — Attack Pattern
- [ ] Implement _attack_behavior() — drones converge on a target point
- [ ] Test with target = center of arena (250, 250)
- [ ] Add a red X marker for the target in run_simulation.py
- [ ] Push commit: "feat: attack convergence behavior"

## Week 5 — Sensor Simulations
- [ ] Create sensors/radar.py — adds Gaussian position noise to true positions
- [ ] Create sensors/optical.py — bounding box detections with false positives
- [ ] Create sensors/rf.py — signal strength model with random dropout
- [ ] Visualize all three sensor outputs alongside true positions
- [ ] Push commit: "feat: radar, optical, RF sensor noise models"

## Week 6 — Polish + Milestone
- [ ] Add configurable params to run_simulation.py (N_DRONES, BEHAVIOR via CLI args)
- [ ] Record a GIF of the simulation (use matplotlib.animation.save or screen record)
- [ ] Add GIF to README.md
- [ ] Write a short "Phase 1 complete" note in ROADMAP.md
- [ ] Tag release: git tag v0.1
- [ ] Push commit: "release: v0.1 Swarm Simulation Engine"

---

## Phase 1 Definition of Done

A recruiter should be able to:
1. Clone the repo
2. Run `pip install numpy matplotlib`
3. Run `python simulator/run_simulation.py`
4. Watch a swarm of drones move in real time

That is Phase 1. Nothing more, nothing less.
