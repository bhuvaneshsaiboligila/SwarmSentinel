# SwarmSentinel — 6-Month Development Roadmap

## Overview

This document is the master plan for building SwarmSentinel from scratch over 28 weeks.
It is designed to grow with your skills — Phase 1 starts at beginner Python level,
Phase 4 requires the skills you will have built along the way.

**Rule:** Never rush to the next phase. A polished Phase 2 is better than a broken Phase 4.

---

## Phase 1 — Swarm Simulator (Weeks 1–6)

**Goal:** You can simulate a drone swarm and visualize it.

### What You Build
- 2D swarm simulator: N drones with configurable behavior
  - Random movement
  - Coordinated swarm (flocking algorithm)
  - Attack pattern (converging on a target)
- Three independent sensor simulations:
  - Radar: position + velocity with Gaussian noise
  - Optical: bounding box detections with false positives
  - RF: signal strength with dropout simulation
- Real-time Matplotlib visualization

### Skills You Learn
- NumPy arrays and vectorized math
- Object-oriented Python (classes, inheritance)
- Basic 2D physics simulation
- Matplotlib animation

### Files to Create
```
simulator/
├── __init__.py
├── drone.py          # Single drone entity
├── swarm.py          # Swarm controller (flocking logic)
├── behaviors.py      # Attack, random, coordinated behaviors
└── run_simulation.py # Entry point

sensors/
├── __init__.py
├── radar.py          # Radar noise model
├── optical.py        # Camera/optical noise model
└── rf.py             # RF signal model
```

### Milestone
`git tag v0.1` — Swarm Simulation Engine  
A GIF of 50 drones moving in coordinated attack pattern in the README.

---

## Phase 2 — Sensor Fusion (Weeks 7–12)

**Goal:** Three noisy sensors become one clean, unified track.

### What You Build
- Kalman Filter for single-target tracking (radar + optical fusion)
- Extended Kalman Filter (EKF) for nonlinear drone movement
- Multi-target track manager:
  - Spawn new tracks when new detections appear
  - Drop tracks when a target disappears
  - Handle occlusion (drone temporarily hidden from sensors)
- Side-by-side visualization: raw noisy data vs. fused clean track

### Skills You Learn
- Kalman filter mathematics (predict/update cycle)
- FilterPy library
- Probabilistic state estimation
- Multi-hypothesis tracking concepts

### Files to Create
```
fusion/
├── __init__.py
├── kalman_filter.py   # Standard KF implementation
├── ekf.py             # Extended KF for nonlinear motion
├── track_manager.py   # Multi-target track lifecycle
└── fusion_pipeline.py # Main pipeline: sensors in, tracks out
```

### Milestone
`git tag v0.2` — Multi-Sensor Fusion Pipeline  
Benchmark: fusion error vs. number of sensors (1, 2, 3).

---

## Phase 3 — Swarm AI (Weeks 13–20)

**Goal:** The system classifies threats and predicts trajectories.

### What You Build
- Synthetic dataset generator:
  - Label 0: Individual drone (random movement)
  - Label 1: Swarm (flocking, coordinated)
  - Label 2: Swarm attack (converging on target)
- PyTorch LSTM classifier on trajectory sequences
- Trajectory predictor: where will this drone/swarm be in T seconds?
- Threat prioritization: rank all active tracks by danger score
- Model card documentation

### Skills You Learn
- PyTorch basics (tensors, datasets, training loop)
- LSTM / sequence modeling
- Synthetic data generation
- ML evaluation (precision, recall, confusion matrix)
- Model documentation

### Files to Create
```
ml/
├── __init__.py
├── dataset.py          # Synthetic data generator + DataLoader
├── model.py            # LSTM classifier architecture
├── train.py            # Training loop
├── evaluate.py         # Metrics + confusion matrix
├── predictor.py        # Trajectory prediction module
├── threat_ranker.py    # Threat prioritization logic
└── MODEL_CARD.md       # What the model does, limitations, data
```

### Milestone
`git tag v0.3` — Swarm Classification Model  
Target: >85% classification accuracy on synthetic test set.

---

## Phase 4 — HPC Layer + C2 Interface (Weeks 21–28)

**Goal:** It looks like a real system, and it scales.

### What You Build
- Replace internal queues with Apache Kafka (sensors become microservices)
- Parallelize fusion pipeline using Python multiprocessing
- MPI-based distributed track processing (multiple nodes)
- Benchmark report: latency at 10 / 100 / 1000 simulated drones
- FastAPI backend serving the real-time threat picture (WebSocket)
- React C2 dashboard:
  - Live 2D map with track positions
  - Track list with threat scores
  - Swarm detection alerts
  - System health panel

### Skills You Learn
- Apache Kafka (producers, consumers, topics)
- Python multiprocessing
- MPI with mpi4py
- FastAPI + WebSockets
- React basics

### Files to Create
```
c2/
├── backend/
│   ├── main.py          # FastAPI app
│   ├── websocket.py     # Real-time track streaming
│   └── threat_api.py    # REST endpoints
└── frontend/
    ├── src/
    │   ├── App.jsx
    │   ├── Map.jsx
    │   └── ThreatPanel.jsx
    └── package.json

hpc/
├── mpi_tracker.py       # Distributed track processing
└── benchmark.py         # Scale benchmarks

docker-compose.yml       # Kafka + backend + frontend together
```

### Milestone
`git tag v1.0` — Full Stack Demo  
Benchmark report in /benchmarks/results.md  
Demo video in README.

---

## Weekly Habit

Every single week, regardless of phase:
1. Write code — even 20 lines is progress
2. Push to GitHub — maintain commit history
3. Update the phase checklist in this file

A GitHub profile with 28 weeks of consistent commits tells a story.
A GitHub profile with one big push tells nothing.

---

## What Success Looks Like at v1.0

- A recruiter at Helsing can clone the repo and run `docker-compose up` and see a live dashboard
- The README has a demo GIF, architecture diagram, and benchmark numbers
- The ML model has a proper model card
- The commit history shows 6 months of real work
- The code has tests and documentation

That is the bar. Build toward it.
