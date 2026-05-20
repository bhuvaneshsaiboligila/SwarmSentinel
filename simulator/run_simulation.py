# run_simulation.py — Entry point for Phase 1
# Run this to see your swarm in action:
#   python simulator/run_simulation.py [--behavior random|flock|attack]

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse

import numpy as np
import matplotlib
if "--no-display" in sys.argv:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from simulator.swarm import Swarm

# ── Configuration (hardware-safe — see CLAUDE.md) ─────────────────────────────
N_DRONES  = 25
AREA      = (100, 100)    # matches Drone.bounds default (wraps at 100)
DT        = 1 / 15        # 15 fps
MAX_STEPS = 200
INTERVAL  = 67            # ms between frames (≈15 fps)
BG        = "#0a0a1a"
# ──────────────────────────────────────────────────────────────────────────────

BEHAVIOR_COLORS = {"random": "#00ff88", "flock": "#00aaff", "attack": "#ff4444"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SwarmSentinel live simulation")
    parser.add_argument(
        "--behavior", "--mode",
        choices=["random", "flock", "attack"],
        default="flock",
        dest="behavior",
        help="Swarm behavior to simulate (default: flock)",
    )
    parser.add_argument(
        "--save",
        metavar="PATH",
        default=None,
        help="Save animation as GIF to PATH instead of opening a window",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=MAX_STEPS,
        help="Number of animation frames (default: 200)",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Headless mode (Agg backend, no window). For CI and smoke tests.",
    )
    args = parser.parse_args()

    target = np.array([50.0, 50.0]) if args.behavior == "attack" else None
    swarm  = Swarm(n_drones=N_DRONES, spawn_area=AREA,
                   behavior=args.behavior, target=target)

    if args.no_display and not args.save:
        for _ in range(args.frames):
            swarm.step(DT)
        print("Headless run complete.")
        sys.exit(0)

    dot_color = BEHAVIOR_COLORS.get(args.behavior, "#00ff88")

    # ── Single subplot (hardware-safe) ─────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 7))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, AREA[0])
    ax.set_ylim(0, AREA[1])
    ax.tick_params(colors="gray")
    for sp in ax.spines.values():
        sp.set_edgecolor("#333355")

    scat  = ax.scatter([], [], c=dot_color, s=14, alpha=0.85, zorder=2)
    if args.behavior == "attack" and target is not None:
        ax.plot(target[0], target[1], "rx", ms=14, mew=2.5, zorder=3)
    title = ax.set_title(
        f"SwarmSentinel — {N_DRONES} drones [{args.behavior}]",
        color="white", fontsize=11,
    )
    info = ax.text(0.02, 0.96, "", transform=ax.transAxes,
                   color="gray", fontsize=9, va="top")

    # Track simulation time independently of frame index so --save starts at t=0
    t_sim = [0.0]

    def init():
        scat.set_offsets(np.empty((0, 2)))
        info.set_text("")
        return scat, info

    def animate(frame):
        swarm.step(DT)
        t_sim[0] += DT
        positions = swarm.get_positions()
        scat.set_offsets(positions)
        info.set_text(f"t={t_sim[0]:.1f}s  alive={swarm.n_alive}")
        return scat, info

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
