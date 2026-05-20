# Model Card — SwarmClassifier

## Model

**Name:** SwarmClassifier  
**Type:** LSTM sequence classifier  
**Task:** 3-class drone swarm threat classification  
**File:** `ml/model.py` — `SwarmClassifier(nn.Module)`

---

## Task

Classify a short trajectory sequence into one of three threat categories:

| Label | Class | Description |
|-------|-------|-------------|
| 0 | individual | Single drone moving randomly — low threat |
| 1 | swarm | 15–25 drones in coordinated flock formation — medium threat |
| 2 | attack | 15–25 drones converging on a target — high threat |

---

## Input

- **Shape:** `(batch, 20, 8)` — 20 timesteps × 8 features per step
- **Timestep:** 1/15 s (15 fps simulation)
- **Sequence length:** 20 steps = ~1.3 seconds of observation

### Features (per timestep)

| # | Name | Description |
|---|------|-------------|
| 0 | `centroid_x` | Mean x-position of radar-detected drones (m) |
| 1 | `centroid_y` | Mean y-position of radar-detected drones (m) |
| 2 | `centroid_vx` | Mean x-velocity of radar-detected drones (m/s) |
| 3 | `centroid_vy` | Mean y-velocity of radar-detected drones (m/s) |
| 4 | `spread_x` | Std of ground-truth drone x-positions (m); 0 for single drone |
| 5 | `spread_y` | Std of ground-truth drone y-positions (m); 0 for single drone |
| 6 | `n_alive_norm` | Active drone count / 25.0 (dimensionless, ≈0.04 for individual) |
| 7 | `mean_speed` | Mean per-drone speed √(vx²+vy²) across alive drones (m/s) |

---

## Architecture

```
LSTM(input=8, hidden=64, layers=2, dropout=0.3, batch_first=True)
  └─ last hidden state (batch, 64)
       └─ Linear(64→32) → ReLU → Dropout(0.3) → Linear(32→3)
```

**Total parameters:** 53,379  
**Output:** raw logits `(batch, 3)` — apply `softmax` for probabilities

---

## Training

| Setting | Value |
|---------|-------|
| Dataset | `SyntheticSwarmDataset` (simulator-generated) |
| Samples | 500 per class × 3 classes = 1,500 total |
| Split | 80% train (1,200) / 20% val (300); seed=42 |
| Epochs | 50 |
| Optimizer | Adam, lr=1e-3 |
| Loss | CrossEntropyLoss |
| Best val acc | 86.7% (epoch 50) |
| Checkpoint | `ml/checkpoints/swarm_classifier.pt` |

---

## Performance

Evaluated on a **held-out test set** (200 samples/class × 3 = 600 total, seed=99, never seen during training).

**Overall accuracy: 86.3% (518/600)**

### Per-class metrics

| Class | Precision | Recall | F1 |
|-------|-----------|--------|----|
| individual | 1.000 | 1.000 | 1.000 |
| swarm | 0.915 | 0.650 | 0.760 |
| attack | 0.729 | 0.940 | 0.821 |
| **macro avg** | **0.881** | **0.863** | **0.860** |

### Confusion matrix

```
                individual   swarm  attack
              ────────────────────────────
  individual │         200       0       0
       swarm │           0     130      70
      attack │           0      12     188
```

**Key observation:** individual is classified perfectly (spread=0 and n_alive_norm≈0.04 are unambiguous). The swarm↔attack confusion (70 swarm→attack, 12 attack→swarm) occurs because an attack formation looks like a flock in the early timesteps of a 1.3-second window before convergence becomes visible.

---

## Limitations

- **Synthetic data only.** Trained and tested entirely on simulator output. Real sensor data will differ in noise characteristics, occlusion patterns, and drone dynamics.
- **Centroid features lose individual drone detail.** Classifying by population centroid + spread cannot distinguish sub-swarm formations, split groups, or partial attacks.
- **Fixed arena (100×100 m), fixed drone count range (15–25).** Performance on larger arenas or different swarm sizes is untested.
- **Short observation window (1.3 s).** Attack vs. swarm confusion drops significantly with longer sequences — the convergence signal takes a few seconds to become unambiguous.
- **No temporal context across sequences.** The model sees each 20-step window independently; it has no memory of what happened before the window began.

---

## Intended Use

**Portfolio demonstration only.** This model illustrates the Phase 3 sensor fusion → AI threat classification pipeline for the SwarmSentinel project. It is not validated for, and must not be used in, any operational counter-drone or defense context.

---

## Files

| File | Description |
|------|-------------|
| `ml/dataset.py` | Synthetic dataset generator |
| `ml/model.py` | Model architecture |
| `ml/train.py` | Training loop |
| `ml/evaluate.py` | Evaluation + confusion matrix |
| `ml/checkpoints/swarm_classifier.pt` | Trained weights (gitignored) |
| `ml/checkpoints/confusion_matrix.txt` | Saved evaluation report |
