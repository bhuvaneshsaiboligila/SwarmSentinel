"""
ml/threat_ranker.py — Threat prioritization for confirmed drone tracks.

Scores each track 0.0–1.0 using a weighted combination of classifier output,
speed, and proximity to arena centre:

    threat_score = 0.5 * attack_prob
                 + 0.3 * speed_norm
                 + 0.2 * proximity_norm

where:
    attack_prob     = softmax probability for class 2 (attack) from SwarmClassifier
    speed_norm      = sequence mean_speed (feature 7 at last timestep) / 10.0, clamped [0,1]
    proximity_norm  = 1 - (dist_to_center / 70.0), clamped [0,1]

Run from project root:
    python ml/threat_ranker.py
"""

import sys
import pathlib

if __name__ == "__main__":
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from ml.model import SwarmClassifier

ARENA_CENTER   = np.array([50.0, 50.0])
MAX_DIST_NORM  = 70.0    # distance at which proximity_norm → 0
MAX_SPEED_NORM = 10.0    # speed at which speed_norm → 1


class ThreatRanker:
    """
    Ranks confirmed tracks by threat score using SwarmClassifier.

    Args:
        model:      a SwarmClassifier instance (eval mode is enforced internally)
        arena_size: (width, height) of the arena; centre is used for proximity
    """

    def __init__(self, model: SwarmClassifier, arena_size: tuple = (100, 100)):
        self.model  = model
        self.model.eval()
        self._center = np.array([arena_size[0] / 2.0, arena_size[1] / 2.0])

    def rank_tracks(
        self,
        tracks: list,
        sequences: list,
    ) -> list:
        """
        Score and rank all confirmed tracks by threat level.

        Args:
            tracks:    list of track dicts from TrackManager.confirmed_tracks():
                       [{track_id, position:[x,y], velocity:[vx,vy], age, missed_frames}]
            sequences: list of (T, 8) numpy arrays, one per track — the feature
                       history used to run the classifier.  Must be same length as tracks.

        Returns:
            list of (track_id, threat_score) tuples sorted descending by score.
        """
        if not tracks:
            return []

        if len(tracks) != len(sequences):
            raise ValueError(
                f"tracks ({len(tracks)}) and sequences ({len(sequences)}) must have the same length"
            )

        # ── Run classifier on all sequences in one batch ───────────────────────
        X = np.stack([s.astype(np.float32) for s in sequences])   # (N, T, 8)
        with torch.no_grad():
            logits = self.model(torch.from_numpy(X))               # (N, 3)
            probs  = torch.softmax(logits, dim=1).numpy()          # (N, 3)

        # ── Score each track ───────────────────────────────────────────────────
        results = []
        for i, (track, seq) in enumerate(zip(tracks, sequences)):
            attack_prob = float(probs[i, 2])

            # mean_speed is feature index 7 at the last observed timestep
            mean_speed  = float(seq[-1, 7])
            speed_norm  = float(np.clip(mean_speed / MAX_SPEED_NORM, 0.0, 1.0))

            pos = np.asarray(track["position"][:2], dtype=float)
            dist_to_center  = float(np.linalg.norm(pos - self._center))
            proximity_norm  = float(np.clip(1.0 - dist_to_center / MAX_DIST_NORM, 0.0, 1.0))

            threat_score = (
                0.5 * attack_prob
                + 0.3 * speed_norm
                + 0.2 * proximity_norm
            )
            results.append((int(track["track_id"]), float(threat_score)))

        results.sort(key=lambda pair: pair[1], reverse=True)
        return results


# ── Smoke test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    CHECKPOINT = pathlib.Path(__file__).parent / "checkpoints" / "swarm_classifier.pt"

    # Load model
    model = SwarmClassifier()
    if CHECKPOINT.exists():
        model.load_state_dict(torch.load(CHECKPOINT, weights_only=True))
        print(f"Loaded checkpoint: {CHECKPOINT}")
    else:
        print("No checkpoint found — using random weights (scores will be noise)")
    model.eval()

    ranker = ThreatRanker(model)

    from ml.dataset import SyntheticSwarmDataset

    # Generate one sequence per class plus two extra to get 5 tracks
    ds = SyntheticSwarmDataset(n_samples_per_class=2, seq_len=20, noise=True, seed=13)
    # ds.X shape: (6, 20, 8), ds.y: [0,0,1,1,2,2]

    # Build 5 fake tracks with varied positions and velocities
    fake_tracks = [
        {"track_id": 0, "position": [50.0, 50.0], "velocity": [0.1, 0.1],  "age": 10, "missed_frames": 0},  # centre, slow
        {"track_id": 1, "position": [10.0, 10.0], "velocity": [3.0, 3.0],  "age": 8,  "missed_frames": 0},  # corner, fast
        {"track_id": 2, "position": [48.0, 52.0], "velocity": [2.5, -2.5], "age": 15, "missed_frames": 0},  # near centre, fast
        {"track_id": 3, "position": [80.0, 80.0], "velocity": [0.5, 0.5],  "age": 5,  "missed_frames": 1},  # far, slow
        {"track_id": 4, "position": [55.0, 45.0], "velocity": [1.5, 1.5],  "age": 12, "missed_frames": 0},  # near centre, mid
    ]

    # Use the first 5 dataset sequences (mix of classes: 0,0,1,1,2)
    sequences = [ds.X[i] for i in range(5)]

    ranked = ranker.rank_tracks(fake_tracks, sequences)

    label_names = {0: "individual", 1: "swarm", 2: "attack"}
    seq_classes = [int(ds.y[i]) for i in range(5)]

    print()
    print(f"{'Rank':<5}  {'Track':>5}  {'Score':>6}  {'Seq class':<12}  {'Position':>18}  {'Speed ft7':>9}")
    print("─" * 68)
    for rank, (tid, score) in enumerate(ranked, 1):
        seq   = sequences[tid]
        cls   = seq_classes[tid]
        pos   = fake_tracks[tid]["position"]
        speed = float(seq[-1, 7])
        print(f"{rank:<5}  {tid:>5}  {score:>6.3f}  {label_names[cls]:<12}  "
              f"[{pos[0]:5.1f}, {pos[1]:5.1f}]  {speed:>9.3f}")

    assert len(ranked) == 5
    scores = [s for _, s in ranked]
    assert scores == sorted(scores, reverse=True), "Output must be sorted descending"
    assert all(0.0 <= s <= 1.0 for _, s in ranked), "Scores must be in [0, 1]"
    print()
    print("Smoke test passed.")
