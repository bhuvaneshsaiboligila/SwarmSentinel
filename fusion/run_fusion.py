"""
fusion/run_fusion.py — Phase 2 visualization: raw sensor data vs. fused tracks.

Two-subplot FuncAnimation (hardware-safe: 2 subplots, 25 drones, 15 fps, 150 frames).
  Left subplot  — Noisy raw detections: radar (blue), optical (green), RF count (orange).
  Right subplot — Kalman-fused track estimates with track ID labels.

Usage:
    python fusion/run_fusion.py
    python fusion/run_fusion.py --save assets/phase2_fusion_demo.gif
"""

import sys
import pathlib
import argparse

if __name__ == "__main__":
    # Allow `python fusion/run_fusion.py` from the project root.
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import matplotlib
if "--no-display" in sys.argv:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from simulator.swarm import Swarm
from fusion.fusion_pipeline import FusionPipeline

# ── Configuration (hardware-safe — see CLAUDE.md) ─────────────────────────────
N_DRONES  = 25
AREA      = (100, 100)
DT        = 1 / 15
MAX_STEPS = 150
INTERVAL  = 67        # ms ≈ 15 fps
BG        = "#0a0a1a"

C_RADAR   = "#4488ff"   # blue
C_OPTICAL = "#44ff88"   # green
C_RF_TEXT = "#ff8844"   # orange  (RF is scalar-only; shown as text, not dots)
C_FUSED   = "#ff4444"   # red
# ──────────────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SwarmSentinel Phase 2 — sensor fusion viz")
    parser.add_argument(
        "--save", metavar="PATH", default=None,
        help="Save animation as GIF instead of opening a window",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=MAX_STEPS,
        help="Number of animation frames (default: 150)",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Headless mode (Agg backend, no window). For CI and smoke tests.",
    )
    parser.add_argument(
        "--ekf",
        action="store_true",
        default=False,
        help="Use DroneEKF (5-state CTR) instead of DroneKalmanFilter (4-state CV).",
    )
    args = parser.parse_args()

    # ── Simulation objects ─────────────────────────────────────────────────────
    swarm    = Swarm(n_drones=N_DRONES, spawn_area=AREA, behavior="flock")
    pipeline = FusionPipeline(swarm, dt=DT, use_ekf=args.ekf)

    if args.no_display and not args.save:
        for _ in range(args.frames):
            swarm.step(DT)
            pipeline.step()
        print("Headless run complete.")
        sys.exit(0)

    # ── Figure: 2 subplots side by side (hardware limit: max 2) ───────────────
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12, 6))
    fig.patch.set_facecolor(BG)

    for ax, title in [(ax_l, "Raw Sensor Detections"), (ax_r, "Fused Tracks")]:
        ax.set_facecolor(BG)
        ax.set_xlim(0, AREA[0])
        ax.set_ylim(0, AREA[1])
        ax.tick_params(colors="gray")
        ax.set_title(title, color="white", fontsize=11)
        for sp in ax.spines.values():
            sp.set_edgecolor("#333355")

    # Left subplot — one scatter artist per sensor that has spatial output
    scat_radar = ax_l.scatter(
        [], [], c=C_RADAR, s=18, alpha=0.85, zorder=3, label="Radar [x,y,vx,vy]",
    )
    scat_optical = ax_l.scatter(
        [], [], c=C_OPTICAL, s=12, alpha=0.75, marker="s", zorder=3, label="Optical [x,y]",
    )
    # RF gives signal strengths only — no spatial position — displayed as text
    rf_ann = ax_l.text(
        0.02, 0.96, "RF: —", transform=ax_l.transAxes,
        color=C_RF_TEXT, fontsize=9, va="top",
    )
    ax_l.legend(
        loc="lower right", fontsize=8,
        facecolor="#111130", edgecolor="#333355", labelcolor="white",
    )
    info_l = ax_l.text(0.02, 0.04, "", transform=ax_l.transAxes, color="gray", fontsize=9)

    # Right subplot — fused tracks
    scat_tracks = ax_r.scatter(
        [], [], c=C_FUSED, s=22, alpha=0.9, zorder=3, label="Fused track",
    )
    ax_r.legend(
        loc="lower right", fontsize=8,
        facecolor="#111130", edgecolor="#333355", labelcolor="white",
    )
    info_r = ax_r.text(0.02, 0.04, "", transform=ax_r.transAxes, color="gray", fontsize=9)

    track_labels = []   # dynamic Text objects for track ID annotations
    t_sim = [0.0]

    def init():
        scat_radar.set_offsets(np.empty((0, 2)))
        scat_optical.set_offsets(np.empty((0, 2)))
        scat_tracks.set_offsets(np.empty((0, 2)))
        info_l.set_text("")
        info_r.set_text("")
        rf_ann.set_text("RF: —")
        return []

    def animate(frame):
        swarm.step(DT)
        t_sim[0] += DT

        # ── Sense + fuse via pipeline ──────────────────────────────────────────
        det          = pipeline.step()
        track_states = pipeline.track_manager.confirmed_tracks()
        radar_det    = det["radar"]    # ndarray (n_detected, 4)  [x, y, vx, vy]
        optical_det  = det["optical"]  # list of {drone_id, bbox, confidence}
        rf_det       = det["rf"]       # ndarray (n_alive,)  signal strengths

        # ── Left subplot: raw detections ───────────────────────────────────────
        if radar_det.shape[0] > 0:
            scat_radar.set_offsets(radar_det[:, :2])
        else:
            scat_radar.set_offsets(np.empty((0, 2)))

        opt_centres = []
        for det in optical_det:
            b = det["bbox"]                               # [x_tl, y_tl, w, h]
            opt_centres.append([b[0] + b[2] / 2.0, b[1] + b[3] / 2.0])

        if opt_centres:
            scat_optical.set_offsets(np.array(opt_centres))
        else:
            scat_optical.set_offsets(np.empty((0, 2)))

        n_rf_alive   = int(len(rf_det))
        n_rf_dropout = int((rf_det == 0.0).sum())
        rf_ann.set_text(f"RF: {n_rf_alive - n_rf_dropout}/{n_rf_alive} signals")

        info_l.set_text(
            f"t={t_sim[0]:.1f}s  "
            f"radar={radar_det.shape[0]}  "
            f"optical={len(opt_centres)}"
        )

        # ── Right subplot: fused tracks ────────────────────────────────────────
        # Remove last frame's track ID labels before drawing new ones
        for lbl in track_labels:
            lbl.remove()
        track_labels.clear()

        if track_states:
            positions = np.array([s["position"] for s in track_states])
            scat_tracks.set_offsets(positions)
            for s in track_states:
                lbl = ax_r.text(
                    s["position"][0] + 1.5, s["position"][1] + 1.5,
                    str(s["track_id"]),
                    color="white", fontsize=7, alpha=0.7, zorder=4,
                )
                track_labels.append(lbl)
        else:
            scat_tracks.set_offsets(np.empty((0, 2)))

        info_r.set_text(f"t={t_sim[0]:.1f}s  tracks={len(track_states)}")
        return []

    ani = animation.FuncAnimation(
        fig, animate, init_func=init,
        frames=args.frames, interval=INTERVAL, blit=False,
    )

    plt.tight_layout()

    if args.save:
        writer = animation.PillowWriter(fps=15)
        print(f"Saving {args.frames} frames to {args.save} …")
        ani.save(args.save, writer=writer)
        print(f"Saved → {args.save}")
    else:
        plt.show()
