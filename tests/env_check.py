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

    # ── Smoke test: run_simulation.py imports + exits cleanly ─────────────────
    import subprocess, tempfile
    with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as tmp:
        smoke_path = tmp.name
    try:
        result = subprocess.run(
            [sys.executable, "simulator/run_simulation.py",
             "--behavior", "random", "--frames", "5", "--save", smoke_path],
            capture_output=True,
        )
        assert result.returncode == 0, (
            f"run_simulation.py smoke test failed (rc={result.returncode}):\n"
            + result.stderr.decode()
        )
    finally:
        os.unlink(smoke_path)

    # ── TEST 1: Phase 1 CLI (--mode alias + --no-display) ─────────────────────
    _cmd1 = [sys.executable, "simulator/run_simulation.py",
             "--mode", "random", "--frames", "5", "--no-display"]
    _r1 = subprocess.run(_cmd1, capture_output=True)
    if _r1.returncode == 0:
        assert "Animation was deleted" not in _r1.stderr.decode(), \
            "TEST 1 produced 'Animation was deleted' warning"
        print("PASS  simulator/run_simulation.py --mode random --frames 5 --no-display")
    else:
        print("FAIL  simulator/run_simulation.py --mode random --frames 5 --no-display")
        raise AssertionError(_r1.stderr.decode())

    # ── TEST 2: Phase 2 CLI (--frames + --no-display) ─────────────────────────
    _cmd2 = [sys.executable, "fusion/run_fusion.py",
             "--frames", "5", "--no-display"]
    _r2 = subprocess.run(_cmd2, capture_output=True)
    if _r2.returncode == 0:
        assert "Animation was deleted" not in _r2.stderr.decode(), \
            "TEST 2 produced 'Animation was deleted' warning"
        print("PASS  fusion/run_fusion.py --frames 5 --no-display")
    else:
        print("FAIL  fusion/run_fusion.py --frames 5 --no-display")
        raise AssertionError(_r2.stderr.decode())

    print("ENV OK")
    sys.exit(0)

except Exception as exc:
    print(f"ENV FAIL: {exc}", file=sys.stderr)
    sys.exit(1)
