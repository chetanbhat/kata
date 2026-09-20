import marimo

__generated_with = "0.11.0"
app = marimo.App(width="medium", app_title="Kata ML Practice Playground")


@app.cell
def _():
    import marimo as mo
    import sys
    import os
    import subprocess
    from pathlib import Path

    # Ensure kata root directory is in sys.path
    kata_dir = Path(__file__).parent.parent if "__file__" in globals() else Path.cwd()
    if str(kata_dir) not in sys.path:
        sys.path.insert(0, str(kata_dir))

    return Path, kata_dir, mo, os, subprocess, sys


@app.cell
def _(mo):
    mo.md(
        r"""
        # 🥋 Kata — Test-Driven ML Practice Playground

        Welcome to **Kata**! Learn ML from raw math to modern architectures by writing code and passing test suites.
        Every exercise enforces three fundamental invariants:
        - 📐 **Shape** — Tensor dimensions scale and match across batches (`B,C,H,W` / `B,T,D`).
        - ⚡ **Gradient Flow** — Parameter `.grad` values are non-zero and non-None after `.backward()`.
        - 📉 **Convergence** — Loss strictly decreases across optimization loops.

        ---
        """
    )
    return


@app.cell
def _(mo):
    run_btn = mo.ui.run_button(label="🚀 Run All Kata Verification Tests")
    impl_choice = mo.ui.dropdown(
        options={"exercises": "Your Code (exercises/)", "solutions": "Reference Solutions (solutions/)"},
        value="exercises",
        label="Implementation Mode:",
    )
    mo.hstack([impl_choice, run_btn], align="center")
    return impl_choice, run_btn


@app.cell
def _(impl_choice, kata_dir, mo, run_btn, subprocess):
    if run_btn.value:
        with mo.status.spinner(title="Running PyTest suites across all modules..."):
            env = dict(os.environ, KATA_IMPL=impl_choice.value)
            cmd = [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"]
            result = subprocess.run(cmd, cwd=str(kata_dir), capture_output=True, text=True, env=env)
            
            output = result.stdout + "\n" + result.stderr
            status_color = "green" if result.returncode == 0 else "red"
            status_msg = "SUCCESS — All Tests Passed!" if result.returncode == 0 else "SOME TESTS FAILING (Expected for uncompleted exercises)"
            
            test_output = mo.md(f"""
            ### Test Results ({impl_choice.value})
            **Status:** <span style="color: {status_color}; font-weight: bold;">{status_msg}</span>

            ```text
            {output[-3000:]}
            ```
            """)
    else:
        test_output = mo.md("*Click **Run All Kata Verification Tests** above to evaluate your progress.*")

    test_output
    return cmd, env, output, result, status_color, status_msg, test_output


@app.cell
def _(mo):
    mo.md(
        r"""
        ---
        ## 📚 Modules & Playgrounds

        Select a playground below to open its dedicated interactive notebook:

        | # | Playground | Topics Covered | Interactive Visualizations |
        |---|------------|----------------|----------------------------|
        | 01 | **[01_foundations_playground.py](./01_foundations_playground.py)** | Linear/Logistic Regression, Perceptron, MLP, Losses | Decision Boundary Plot, Loss Surface |
        | 02 | **[02_deep_learning_playground.py](./02_deep_learning_playground.py)** | Autograd Engine, BatchNorm, LayerNorm, Conv2D | Computation Graph, Normalization Animations |
        | 03 | **[03_training_algos_playground.py](./03_training_algos_playground.py)** | SGD + Momentum, Adam, Cosine Schedulers, Loops | Optimizer Trajectory on 2D Loss Surfaces |
        | 04 | **[04_architectures_playground.py](./04_architectures_playground.py)** | ResNet, LSTM, Transformer (SDPA, RoPE, MHA) | Multi-Head Attention Heatmaps, RoPE Rotations |
        | 09 | **[09_capstone_playground.py](./09_capstone_playground.py)** | Two-moons End-to-End Classification Capstone | Live Epoch Training & Overfitting Diagnosis |

        ---
        ### 💡 How to Practice
        1. Open any playground notebook.
        2. Read the mathematical intuition and inspect the code stub.
        3. Fill in the `# TODO` markers.
        4. Click the **Run Module Tests** button in the notebook to instantly verify your code!
        """
    )
    return


if __name__ == "__main__":
    app.run()
