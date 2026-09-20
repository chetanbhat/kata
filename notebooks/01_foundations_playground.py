import marimo

__generated_with = "0.11.0"
app = marimo.App(width="medium", app_title="Kata 01: Foundations Playground")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    import sys
    import subprocess
    from pathlib import Path

    # Ensure kata root is in sys.path
    kata_dir = Path(__file__).parent.parent if "__file__" in globals() else Path.cwd()
    if str(kata_dir) not in sys.path:
        sys.path.insert(0, str(kata_dir))

    from importlib import import_module
    
    # Try importing student exercises or solutions
    try:
        foundations = import_module("exercises.01_foundations.perceptron_mlp")
        losses_mod = import_module("exercises.01_foundations.losses")
        linear_mod = import_module("exercises.01_foundations.linear_logistic")
        impl_source = "exercises"
    except Exception:
        foundations = import_module("solutions.01_foundations.perceptron_mlp")
        losses_mod = import_module("solutions.01_foundations.losses")
        linear_mod = import_module("solutions.01_foundations.linear_logistic")
        impl_source = "solutions"

    return (
        Path,
        foundations,
        import_module,
        impl_source,
        kata_dir,
        linear_mod,
        losses_mod,
        mo,
        np,
        plt,
        subprocess,
        sys,
    )


@app.cell
def _(impl_source, mo):
    mo.md(
        f"""
        # 📐 Module 01: Foundations Playground
        *Currently using implementation from: `{impl_source}/`*

        In this module, you build foundational ML primitives from raw NumPy math:
        1. **Linear & Logistic Regression** (OLS closed form, L1/L2 regularized gradient descent).
        2. **Perceptron & MLP** (Rosenblatt algorithm, manual backpropagation).
        3. **Loss Functions** (MSE, numerically stable BCE & Cross-Entropy).

        ---
        """
    )
    return


@app.cell
def _(mo):
    mo.md("### 🧪 1. Activations & Derivatives Visualizer")
    act_choice = mo.ui.dropdown(
        options=["ReLU", "Sigmoid", "Tanh", "GELU"],
        value="ReLU",
        label="Select Activation Function:"
    )
    act_choice
    return (act_choice,)


@app.cell
def _(act_choice, foundations, mo, np, plt):
    x = np.linspace(-4, 4, 300)
    fig, ax = plt.subplots(figsize=(7, 3.5), dpi=120)

    try:
        if act_choice.value == "ReLU":
            y = foundations.relu(x)
            dy = foundations.relu_deriv(x)
        elif act_choice.value == "Sigmoid":
            y = foundations.sigmoid(x)
            dy = foundations.sigmoid_deriv(x)
        elif act_choice.value == "Tanh":
            y = foundations.tanh(x)
            dy = foundations.tanh_deriv(x)
        elif act_choice.value == "GELU":
            y = foundations.gelu(x)
            dy = foundations.gelu_deriv(x)

        ax.plot(x, y, label=f"{act_choice.value}(x)", lw=2, color="#1f77b4")
        ax.plot(x, dy, label=f"d/dx {act_choice.value}(x)", lw=2, linestyle="--", color="#ff7f0e")
        ax.axhline(0, color="gray", linewidth=0.8, linestyle=":")
        ax.axvline(0, color="gray", linewidth=0.8, linestyle=":")
        ax.set_title(f"Activation & Derivative: {act_choice.value}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plot_out = mo.pyplot(fig)
    except NotImplementedError:
        plot_out = mo.md(f"⚠️ `{act_choice.value}` is not implemented yet in `exercises/01_foundations/perceptron_mlp.py`!")
    except Exception as e:
        plot_out = mo.md(f"❌ Error evaluating `{act_choice.value}`: `{e}`")

    plt.close(fig)
    plot_out
    return ax, dy, e, fig, plot_out, x, y


@app.cell
def _(mo):
    mo.md("### ⚡ 2. Perceptron Decision Boundary Explorer")
    n_samples = mo.ui.slider(start=20, stop=200, step=10, value=50, label="Number of points:")
    lr_slider = mo.ui.slider(start=0.01, stop=0.5, step=0.01, value=0.1, label="Learning Rate:")
    epochs_slider = mo.ui.slider(start=5, stop=100, step=5, value=30, label="Epochs:")
    mo.hstack([n_samples, lr_slider, epochs_slider], align="center")
    return epochs_slider, lr_slider, n_samples


@app.cell
def _(epochs_slider, foundations, lr_slider, mo, n_samples, np, plt):
    rng = np.random.default_rng(42)
    # Generate linearly separable 2D dataset
    X_pos = rng.normal(loc=[1.5, 1.5], scale=0.8, size=(n_samples.value // 2, 2))
    X_neg = rng.normal(loc=[-1.5, -1.5], scale=0.8, size=(n_samples.value // 2, 2))
    X_data = np.vstack([X_pos, X_neg])
    y_data = np.hstack([np.ones(n_samples.value // 2), np.zeros(n_samples.value // 2)])

    fig, ax = plt.subplots(figsize=(7, 4), dpi=120)
    ax.scatter(X_pos[:, 0], X_pos[:, 1], color="blue", label="Class 1", alpha=0.7)
    ax.scatter(X_neg[:, 0], X_neg[:, 1], color="red", label="Class 0", alpha=0.7)

    try:
        p = foundations.Perceptron(n_features=2, lr=lr_slider.value, seed=42)
        p.fit(X_data, y_data, epochs=epochs_slider.value)

        # Plot decision boundary: w0*x0 + w1*x1 + b = 0 => x1 = -(w0*x0 + b)/w1
        x_vals = np.linspace(-4, 4, 100)
        if abs(p.w[1]) > 1e-5:
            y_vals = -(p.w[0] * x_vals + p.b) / p.w[1]
            ax.plot(x_vals, y_vals, "g-", linewidth=2, label="Learned Boundary")
            ax.set_ylim(-4, 4)

        ax.set_title(f"Perceptron Boundary (LR={lr_slider.value}, Epochs={epochs_slider.value})")
        ax.legend()
        ax.grid(True, alpha=0.3)
        perc_out = mo.pyplot(fig)
    except NotImplementedError:
        perc_out = mo.md("⚠️ `Perceptron` is not implemented yet in `exercises/01_foundations/perceptron_mlp.py`!")
    except Exception as e:
        perc_out = mo.md(f"❌ Error fitting Perceptron: `{e}`")

    plt.close(fig)
    perc_out
    return X_data, X_neg, X_pos, ax, e, fig, p, perc_out, rng, x_vals, y_data, y_vals


@app.cell
def _(mo):
    mo.md(
        r"""
        ---
        ### 🚀 3. Run Module 01 Tests
        Verify your implementation of `linear_logistic.py`, `perceptron_mlp.py`, and `losses.py`:
        """
    )
    test_btn = mo.ui.run_button(label="Run Module 01 PyTest Suite")
    test_btn
    return (test_btn,)


@app.cell
def _(kata_dir, mo, subprocess, sys, test_btn):
    if test_btn.value:
        with mo.status.spinner(title="Running pytest tests/test_01_foundations.py..."):
            cmd = [sys.executable, "-m", "pytest", "tests/test_01_foundations.py", "-v"]
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
