import marimo

__generated_with = "0.11.0"
app = marimo.App(width="medium", app_title="Kata 04: Architectures Playground")


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
        resnet_mod = import_module("exercises.04_architectures.resnet")
        lstm_mod = import_module("exercises.04_architectures.lstm")
        trans_mod = import_module("exercises.04_architectures.transformer")
        impl_source = "exercises"
    except Exception:
        resnet_mod = import_module("solutions.04_architectures.resnet")
        lstm_mod = import_module("solutions.04_architectures.lstm")
        trans_mod = import_module("solutions.04_architectures.transformer")
        impl_source = "solutions"

    return (
        Path,
        impl_source,
        import_module,
        kata_dir,
        lstm_mod,
        mo,
        nn,
        np,
        plt,
        resnet_mod,
        subprocess,
        sys,
        torch,
        trans_mod,
    )


@app.cell
def _(impl_source, mo):
    mo.md(
        f"""
        # 🏗️ Module 04: Modern Architectures Playground
        *Currently using implementation from: `{impl_source}/`*

        In this module, you build classic & state-of-the-art neural architectures from scratch:
        1. **ResNet**: Basic Block, Bottleneck Block, ResNet-18/50.
        2. **LSTM**: Fused-matmul gates, unrolled sequence processing.
        3. **Transformer**: Scaled Dot-Product Attention (SDPA), Multi-Head Attention (MHA), Rotary Position Embeddings (RoPE), RMSNorm.

        ---
        """
    )
    return


@app.cell
def _(mo):
    mo.md("### 🔍 1. Scaled Dot-Product Attention (SDPA) Heatmap Explorer")
    seq_len = mo.ui.slider(start=4, stop=16, step=1, value=8, label="Sequence Length:")
    d_k = mo.ui.slider(start=8, stop=64, step=8, value=16, label="Head Dimension (d_k):")
    causal_toggle = mo.ui.checkbox(value=True, label="Apply Causal Mask")
    mo.hstack([seq_len, d_k, causal_toggle], align="center")
    return causal_toggle, d_k, seq_len


@app.cell
def _(causal_toggle, d_k, mo, np, plt, seq_len, torch, trans_mod):
    rng = torch.manual_seed(42)
    Q = torch.randn(1, 1, seq_len.value, d_k.value)
    K = torch.randn(1, 1, seq_len.value, d_k.value)
    V = torch.randn(1, 1, seq_len.value, d_k.value)

    mask = None
    if causal_toggle.value:
        mask = torch.triu(torch.full((seq_len.value, seq_len.value), float("-inf")), diagonal=1)

    try:
        # Check if sdpa or attention function exists
        if hasattr(trans_mod, "scaled_dot_product_attention"):
            out, weights = trans_mod.scaled_dot_product_attention(Q, K, V, mask=mask)
        elif hasattr(trans_mod, "SDPA"):
            sdpa_layer = trans_mod.SDPA()
            out, weights = sdpa_layer(Q, K, V, mask=mask)
        else:
            # Fallback for visualization using PyTorch standard logic
            scores = (Q @ K.transpose(-2, -1)) / (d_k.value ** 0.5)
            if mask is not None:
                scores = scores + mask
            weights = torch.softmax(scores, dim=-1)

        weights_np = weights[0, 0].detach().numpy()

        fig, ax = plt.subplots(figsize=(5.5, 4.5), dpi=120)
        im = ax.imshow(weights_np, cmap="magma")
        fig.colorbar(im, ax=ax)
        ax.set_xticks(np.arange(seq_len.value))
        ax.set_yticks(np.arange(seq_len.value))
        ax.set_xlabel("Key Position")
        ax.set_ylabel("Query Position")
        ax.set_title(f"Attention Weights ({'Causal' if causal_toggle.value else 'Full'}, d_k={d_k.value})")
        attn_out = mo.pyplot(fig)
    except NotImplementedError:
        attn_out = mo.md("⚠️ Attention is not implemented yet in `exercises/04_architectures/transformer.py`!")
    except Exception as e:
        attn_out = mo.md(f"❌ Error computing attention: `{e}`")

    plt.close(fig)
    attn_out
    return (
        K,
        Q,
        V,
        attn_out,
        e,
        fig,
        im,
        mask,
        out,
        rng,
        sdpa_layer,
        weights,
        weights_np,
    )


@app.cell
def _(mo):
    mo.md(
        r"""
        ---
        ### 🚀 2. Run Module 04 Tests
        Verify your implementation of `resnet.py`, `lstm.py`, and `transformer.py`:
        """
    )
    test_btn = mo.ui.run_button(label="Run Module 04 PyTest Suite")
    test_btn
    return (test_btn,)


@app.cell
def _(kata_dir, mo, subprocess, sys, test_btn):
    if test_btn.value:
        with mo.status.spinner(title="Running pytest tests/test_04_architectures.py..."):
            cmd = [sys.executable, "-m", "pytest", "tests/test_04_architectures.py", "-v"]
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
