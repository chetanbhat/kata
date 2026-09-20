# Kata — Test-Driven ML Practice Workspace

Build ML from raw math to modern architectures through code. Every exercise
has a stub (`exercises/`), a reference implementation (`solutions/`), and a
pytest suite (`tests/`) enforcing three invariants:

- **S**hape — tensor dims scale and match across batches (`B,C,H,W` / `B,T,D`).
- **G**radient flow — after `.backward()`, every parameter has a non-zero,
  non-`None` `.grad` (NumPy modules use finite-difference probes instead).
- **C**onvergence — a 10-step mock optimization loop strictly decreases loss.

## Setup

```bash
export UV_CACHE_DIR=/tmp/uv-cache   # only if the shared uv cache is read-only
uv venv
uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv/bin/python numpy pytest matplotlib
```

`requirements.txt` pins the same set (`torch` via the CPU index).

## How to practice

1. Pick an exercise, implement the `# TODO` markers in `exercises/...`.
2. Run its test file from the table (import it the way the tests do, e.g.
   `from importlib import import_module; m = import_module("exercises.01_foundations.losses")`
   — the leading-digit dirs need `import_module`, not plain `import`).
3. Compare against `solutions/...` when stuck.

```bash
.venv/bin/python -m pytest tests/ -q            # whole suite (125 tests)
.venv/bin/python -m pytest tests/test_01_foundations.py -q
```

## Tracker

| # | Exercise | Core math objective | Invariants | Run |
|---|----------|---------------------|------------|-----|
| 1.1 | `exercises/01_foundations/linear_logistic.py` | OLS closed form `(X'X+λI)^{-1}X'y`; vectorized L1/L2 logistic GD | S, G(fd), C | `pytest tests/test_01_foundations.py -k "ols or ridge or logistic or sigmoid"` |
| 1.2 | `exercises/01_foundations/perceptron_mlp.py` | Rosenblatt updates; backprop with ReLU/GeLU/Sigmoid/Tanh | S, G(fd), C | `pytest tests/test_01_foundations.py -k "perceptron or mlp or activation or numerical"` |
| 1.3 | `exercises/01_foundations/losses.py` | MSE / clipped BCE / log-sum-exp cross-entropy | S, stability | `pytest tests/test_01_foundations.py -k "mse or bce or cross_entropy or stable"` |
| 2.1 | `exercises/02_deep_learning/autograd.py` | Dynamic graph + chain rule in reverse-topological order | G(exact), C | `pytest tests/test_02_deep_learning.py -k autograd` |
| 2.2 | `exercises/02_deep_learning/layers.py` | He/Xavier init; running-stat BatchNorm; LayerNorm; inverted Dropout | S, G, C | `pytest tests/test_02_deep_learning.py -k "linear or norm or dropout"` |
| 2.3 | `exercises/02_deep_learning/conv2d.py` | Unfold + matmul forward; stride/pad/dilation; exact match to `nn.Conv2d` | S, G, C | `pytest tests/test_02_deep_learning.py -k conv` |
| 3.1 | `exercises/03_training_algos/optimizers.py` | SGD momentum/Nesterov; Adam moments + bias correction | S, G, C | `pytest tests/test_03_training.py -k "sgd or adam or zero_grad"` |
| 3.2 | `exercises/03_training_algos/schedulers.py` | Cosine restarts (`T_mult` growth); warmup→stable→cosine-decay | schedule shape | `pytest tests/test_03_training.py -k "cosine or warmup"` |
| 3.3 | `exercises/03_training_algos/training_loop.py` | Clipping by norm; accumulation; AMP hooks (CPU no-op); scheduler stepping | C, finiteness | `pytest tests/test_03_training.py -k "train_loop"` |
| 4.1 | `exercises/04_architectures/resnet.py` | Post-activation basic block; pre/post bottleneck; 1×1 projection | S, G, C | `pytest tests/test_04_architectures.py -k "residual or bottleneck"` |
| 4.2 | `exercises/04_architectures/lstm.py` | f/i/o/g gates from fused matmuls; unrolled sequence layer | S, G, C | `pytest tests/test_04_architectures.py -k lstm` |
| 4.3 | `exercises/04_architectures/transformer.py` | Masked SDPA; causal MHA; RMSNorm; RoPE; pre-norm decoder block | S, G, C (+causality, norm-preservation) | `pytest tests/test_04_architectures.py -k "attention or mha or rmsnorm or rope or decoder"` |

`G(fd)` = finite-difference gradient probe (NumPy has no autograd).
`tests/test_00_stubs.py` verifies every stub raises `NotImplementedError`.

## Advanced track (Modules 5–8): history of wrong turns

Each module studies a lineage — what was tried, what failed, what replaced
it — with variations side by side so you feel *why* the fix exists. The
running notes live in [RESEARCH_LOG.md](RESEARCH_LOG.md).

| # | Exercise | Variations tracked | Run |
|---|----------|--------------------|-----|
| 5.1 | `exercises/05_optimization/second_order.py` | Damped Newton vs L-BFGS (memory, line search) | `pytest tests/test_05_optimization.py -k "newton or lbfgs"` |
| 5.2 | `exercises/05_optimization/adaptive_history.py` | AdaGrad → RMSprop → AdamW (decoupled) → Lion → Lookahead wrapper | `pytest tests/test_05_optimization.py -k "adagrad or rmsprop or adamw or lion or lookahead"` |
| 5.3 | `exercises/05_optimization/sharpness.py` | SAM perturb-then-update vs SWA iterate averaging | `pytest tests/test_05_optimization.py -k "sam or swa"` |
| 6.1 | `exercises/06_regularization/regularizers.py` | Dropout → DropConnect → Stochastic Depth; label smoothing; Mixup | `pytest tests/test_06_regularization.py -k "dropconnect or stochastic or smoothing or mixup"` |
| 6.2 | `exercises/06_regularization/normalization.py` | BatchNorm story → WeightNorm → SpectralNorm; clipping → gradient penalty | `pytest tests/test_06_regularization.py -k "weightnorm or spectral or penalty"` |
| 7.1 | `exercises/07_sequences/rnn_history.py` | Elman (vanishing, measured) → LSTM (mod. 4) → GRU | `pytest tests/test_07_sequences.py -k "rnn or gru or vanishing"` |
| 7.2 | `exercises/07_sequences/attention_evolution.py` | Absolute → ALiBi; MHA → MQA → GQA; sliding window | `pytest tests/test_07_sequences.py -k "alibi or window or gqa"` |
| 7.3 | `exercises/07_sequences/ssm.py` | ZOH discretization; fixed SSM → selective (Mamba-lite) scan | `pytest tests/test_07_sequences.py -k "ssm or discretize"` |
| 8.1 | `exercises/08_generative/vae.py` | ELBO + beta annealing knob (posterior-collapse handle) | `pytest tests/test_08_generative.py -k "vae or kl or elbo"` |
| 8.2 | `exercises/08_generative/gan.py` | Minimax → non-saturating; WGAN-GP; spectral-normed D | `pytest tests/test_08_generative.py -k "gan or wgan or spectral_normalize"` |
| 8.3 | `exercises/08_generative/diffusion.py` | Linear vs cosine schedule; DDPM vs DDIM; classifier-free guidance | `pytest tests/test_08_generative.py -k "diffusion or ddpm or ddim or guidance"` |

## Layout

```text
kata/
├── exercises/  01_foundations  02_deep_learning  03_training_algos  04_architectures
│               05_optimization  06_regularization  07_sequences  08_generative
├── solutions/  (mirrors exercises, complete implementations)
├── tests/      test_00_stubs  test_01_foundations  test_02_deep_learning
│               test_03_training  test_04_architectures  test_05_optimization
│               test_06_regularization  test_07_sequences  test_08_generative
│               (+ conftest.py)
├── requirements.txt
├── RESEARCH_LOG.md  (cumulative wrong-turns / learnings journal)
└── README.md
```
