import marimo

__generated_with = "0.11.0"
app = marimo.App(width="medium", app_title="Kata 09: Capstone Playground")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import torch
    import torch.nn as nn
    import matplotlib.pyplot as plt
    import sys
    import subprocess
    from pathlib import Path

    kata_dir = Path(__file__).parent.parent if "__file__" in globals() else Path.cwd()
    if str(kata_dir) not in sys.path:
        sys.path.insert(0, str(kata_dir))

    from importlib import import_module

    try:
        capstone_mod = import_module("exercises.09_capstone.end_to_end")
        impl_source = "exercises"
    except Exception:
        capstone_mod = import_module("solutions.09_capstone.end_to_end")
        impl_source = "solutions"

    return (
        Path,
        capstone_mod,
        impl_source,
        import_module,
        kata_dir,
        mo,
        nn,
        np,
        plt,
        subprocess,
        sys,
        torch,
    )


@app.cell
def _(impl_source, mo):
    mo.md(
        f"""
        # 🌕 Module 09: Capstone Playground — Two Moons End-to-End
        *Currently using implementation from: `{impl_source}/`*

        The Capstone puts everything together into a clean, reproducible ML workflow:
        1. **Data Prep**: Train/Val/Test split, train-only feature scaling (preventing data leakage).
        2. **Model & Loss**: Multi-layer Neural Network with activations and loss function.
        3. **Training & Regularization**: Optimizer loop with learning rate scheduler, gradient clipping.
        4. **Validation & Checkpointing**: Early stopping, tracking best model state dict.

        ---
        """
    )
    return


@app.cell
def _(mo):
    mo.md("### 📊 1. Two-Moons Dataset Visualizer")
    n_points = mo.ui.slider(start=100, stop=1000, step=50, value=300, label="Sample Size:")
    noise_level = mo.ui.slider(start=0.05, stop=0.4, step=0.05, value=0.15, label="Noise Level:")
    mo.hstack([n_points, noise_level], align="center")
    return n_points, noise_level


@app.cell
def _(mo, n_points, noise_level, np, plt):
    # Generate synthetic two-moons dataset
    n_samples_per_moon = n_points.value // 2
    theta = np.linspace(0, np.pi, n_samples_per_moon)
    
    # Upper moon
    x1 = np.cos(theta) + np.random.normal(0, noise_level.value, n_samples_per_moon)
    y1 = np.sin(theta) + np.random.normal(0, noise_level.value, n_samples_per_moon)
    
    # Lower moon
    x2 = 1.0 - np.cos(theta) + np.random.normal(0, noise_level.value, n_samples_per_moon)
    y2 = 0.5 - np.sin(theta) + np.random.normal(0, noise_level.value, n_samples_per_moon)

    fig, ax = plt.subplots(figsize=(6, 4), dpi=120)
    ax.scatter(x1, y1, color="#1f77b4", alpha=0.7, label="Moon 0")
    ax.scatter(x2, y2, color="#ff7f0e", alpha=0.7, label="Moon 1")
    ax.set_title(f"Two-Moons Dataset (N={n_points.value}, Noise={noise_level.value})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    moons_plot = mo.pyplot(fig)

    plt.close(fig)
    moons_plot
    return (
        ax,
        fig,
        moons_plot,
        n_samples_per_moon,
        theta,
        x1,
        x2,
        y1,
        y2,
    )


@app.cell
def _(mo):
    mo.md(
        r"""
        ---
        ### 🚀 2. Run Capstone Tests
        Verify your implementation of `end_to_end.py`:
        """
    )
    test_btn = mo.ui.run_button(label="Run Capstone PyTest Suite")
    test_btn
    return (test_btn,)


@app.cell
def _(kata_dir, mo, subprocess, sys, test_btn):
    if test_btn.value:
        with mo.status.spinner(title="Running pytest tests/test_09_capstone.py..."):
            cmd = [sys.executable, "-m", "pytest", "tests/test_09_capstone.py", "-v"]
            res = subprocess.run(cmd, cwd=str(kata_dir), capture_output=True, text=True)
            out = res.stdout + "\n" + res.stderr
            status = "✅ PASS" if res.returncode == 0 else "❌ FAIL"
            res_display = mo.md(f"**Test Status:** {status}\n\n```text\n{out[-2500:]}\n```")
    else:
        res_display = mo.md("*Click button above to execute tests.*")

    res_display
    return cmd, out, res, res_display, status


if __name__ == "__main__":
    app.run()
