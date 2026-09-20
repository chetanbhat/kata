import marimo

__generated_with = "0.11.0"
app = marimo.App(width="medium", app_title="Kata 03: Training Algorithms Playground")


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
        opts_mod = import_module("exercises.03_training_algos.optimizers")
        sched_mod = import_module("exercises.03_training_algos.schedulers")
        loop_mod = import_module("exercises.03_training_algos.training_loop")
        impl_source = "exercises"
    except Exception:
        opts_mod = import_module("solutions.03_training_algos.optimizers")
        sched_mod = import_module("solutions.03_training_algos.schedulers")
        loop_mod = import_module("solutions.03_training_algos.training_loop")
        impl_source = "solutions"

    return (
        Path,
        impl_source,
        import_module,
        kata_dir,
        loop_mod,
        mo,
        nn,
        np,
        opts_mod,
        plt,
        sched_mod,
        subprocess,
        sys,
        torch,
    )


@app.cell
def _(impl_source, mo):
    mo.md(
        f"""
        # ⚙️ Module 03: Training Algorithms Playground
        *Currently using implementation from: `{impl_source}/`*

        In this module, you build core optimization & scheduling components:
        1. **Optimizers**: `SGDMomentum` (with Nesterov option), `Adam` (with bias correction).
        2. **Schedulers**: `CosineAnnealingLR` (with $T_{{mult}}$ restarts), `WarmupStableCosineScheduler`.
        3. **Training Loop**: Gradient clipping by norm, gradient accumulation, AMP hooks, evaluation loops.

        ---
        """
    )
    return


@app.cell
def _(mo):
    mo.md("### 📉 1. Optimizer Trajectory on 2D Quadratic Bowl")
    lr = mo.ui.slider(start=0.01, stop=0.3, step=0.01, value=0.05, label="Learning Rate:")
    steps = mo.ui.slider(start=10, stop=100, step=5, value=40, label="Optimization Steps:")
    opt_choice = mo.ui.dropdown(
        options=["SGD", "SGD+Momentum", "Adam"],
        value="Adam",
        label="Optimizer:"
    )
    mo.hstack([opt_choice, lr, steps], align="center")
    return lr, opt_choice, steps


@app.cell
def _(lr, mo, np, opt_choice, opts_mod, plt, steps, torch):
    # Quadratic bowl: f(x, y) = 0.5 * x^2 + 5 * y^2 (anisotropic surface)
    X_grid, Y_grid = np.meshgrid(np.linspace(-3, 3, 100), np.linspace(-3, 3, 100))
    Z_grid = 0.5 * X_grid**2 + 5.0 * Y_grid**2

    fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=120)
    cs = ax.contour(X_grid, Y_grid, Z_grid, levels=20, cmap="viridis", alpha=0.6)
    ax.clabel(cs, inline=1, fontsize=8)

    # Initial parameter tensor
    w = torch.tensor([-2.5, 2.5], requires_grad=True)

    try:
        if opt_choice.value == "SGD":
            optimizer = torch.optim.SGD([w], lr=lr.value)
        elif opt_choice.value == "SGD+Momentum":
            optimizer = opts_mod.SGDMomentum([w], lr=lr.value, momentum=0.9)
        elif opt_choice.value == "Adam":
            optimizer = opts_mod.Adam([w], lr=lr.value)

        traj = [w.detach().clone().numpy()]

        for i in range(steps.value):
            optimizer.zero_grad()
            loss = 0.5 * w[0]**2 + 5.0 * w[1]**2
            loss.backward()
            optimizer.step()
            traj.append(w.detach().clone().numpy())

        traj = np.array(traj)
        ax.plot(traj[:, 0], traj[:, 1], "ro-", lw=1.5, ms=4, label=f"{opt_choice.value} trajectory")
        ax.plot(traj[0, 0], traj[0, 1], "go", ms=8, label="Start (-2.5, 2.5)")
        ax.plot(0, 0, "r*", ms=10, label="Optimum (0, 0)")
        ax.set_title(f"2D Optimization Trajectory — {opt_choice.value} (LR={lr.value})")
        ax.legend()
        ax.grid(True, alpha=0.3)
        opt_out = mo.pyplot(fig)
    except NotImplementedError:
        opt_out = mo.md(f"⚠️ `{opt_choice.value}` is not implemented yet in `exercises/03_training_algos/optimizers.py`!")
    except Exception as e:
        opt_out = mo.md(f"❌ Error running optimizer: `{e}`")

    plt.close(fig)
    opt_out
    return Z_grid, X_grid, Y_grid, ax, cs, e, fig, i, loss, optimizer, opt_out, traj, w


@app.cell
def _(mo):
    mo.md(
        r"""
        ---
        ### 🚀 2. Run Module 03 Tests
        Verify your implementation of `optimizers.py`, `schedulers.py`, and `training_loop.py`:
        """
    )
    test_btn = mo.ui.run_button(label="Run Module 03 PyTest Suite")
    test_btn
    return (test_btn,)


@app.cell
def _(kata_dir, mo, subprocess, sys, test_btn):
    if test_btn.value:
        with mo.status.spinner(title="Running pytest tests/test_03_training.py..."):
            cmd = [sys.executable, "-m", "pytest", "tests/test_03_training.py", "-v"]
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
