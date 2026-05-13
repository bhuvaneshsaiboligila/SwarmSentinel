"""
tests/env_check.py — Phase 1 environment validation
Run from project root:
    python tests/env_check.py
Prints ENV OK on success, error details on failure.
"""
import sys

try:
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")   # non-interactive; works without a display
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    print(f"python     : {sys.version.split()[0]}")
    print(f"numpy      : {np.__version__}")
    print(f"matplotlib : {matplotlib.__version__}")

    # ── Plot 25 random points ──────────────────────────────────────────────────
    rng = np.random.default_rng(42)
    points = rng.uniform(0, 100, size=(25, 2))

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_facecolor("#0a0a1a")
    fig.patch.set_facecolor("#0a0a1a")
    scat = ax.scatter(points[:, 0], points[:, 1], c="#00aaff", s=18)
    ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    ax.set_title("env_check", color="white")

    # ── FuncAnimation: 30 frames, interval=67 ms ──────────────────────────────
    def animate(frame):
        noise = rng.normal(0, 0.5, size=(25, 2))
        scat.set_offsets(points + noise)
        return (scat,)

    ani = animation.FuncAnimation(
        fig, animate, frames=30, interval=67, blit=True
    )

    # Render all 30 frames via PillowWriter to a temp file (no display needed)
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        ani.save(tmp_path, writer=animation.PillowWriter(fps=15))
        size = os.path.getsize(tmp_path)
        assert size > 1000, f"GIF too small ({size} bytes) — frames may be static"
    finally:
        os.unlink(tmp_path)
    plt.close(fig)

    print("ENV OK")
    sys.exit(0)

except Exception as exc:
    print(f"ENV FAIL: {exc}", file=sys.stderr)
    sys.exit(1)
