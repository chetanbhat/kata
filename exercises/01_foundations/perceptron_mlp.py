"""EXERCISE 1.2 — Perceptron & MLP with manual gradients (NumPy only).

Prereqs: guides/01_prereqs_math.md §1–2 (matmul shapes, chain rule).
Intuition: an MLP is alternating linear maps and fixed nonlinearities; the
loss depends on early weights only *through* later layers, so gradients flow
backwards reusing cached intermediate values. No autograd — you are autograd.

Worked micro-example (do this on paper first):
  X = [[2.0]], y = [[4.0]], one hidden unit, ReLU, W1 = [[3.0]], W2 = [[1.0]].
  Forward: Z1 = 6, A1 = 6, out = 6, L = (6-4)^2 = 4.
  Backward: dL/dout = 2(6-4) = 4; dW2 = A1 * 4 = 24; dA1 = W2 * 4 = 4;
            dZ1 = dA1 * ReLU'(6) = 4; dW1 = X * dZ1 = 8.
  If you can reproduce dW1 = 8, dW2 = 24, you understand backprop.

Shape diagram (memorize this; every layer_tp does the same dance):
  A_prev: (B, d_in) @ W: (d_in, d_out) + b: (d_out,) -> Z: (B, d_out)

Run: pytest tests/test_01_foundations.py -k "perceptron or mlp or activation"
Reading: readings/01_foundations/perceptron_mlp.md
"""

import numpy as np


def relu(z):
    # TODO: Elementwise max(0, z). HINT: np.maximum.
    raise NotImplementedError("# TODO: Implement this")


def relu_deriv(z):
    # TODO: 1 where z > 0 else 0. HINT: boolean mask .astype(float).
    # NOTE: undefined at exactly 0; the convention (z > 0) is standard.
    raise NotImplementedError("# TODO: Implement this")


def sigmoid(z):
    # TODO: 1 / (1 + exp(-z)). HINT: clip z to [-500, 500] first.
    raise NotImplementedError("# TODO: Implement this")


def sigmoid_deriv(z):
    # TODO: s * (1 - s) where s = sigmoid(z). Derive it once on paper!
    raise NotImplementedError("# TODO: Implement this")


def tanh(z):
    # TODO: np.tanh. HINT: this one is given away; check the pattern.
    raise NotImplementedError("# TODO: Implement this")


def tanh_deriv(z):
    # TODO: 1 - tanh(z)^2. HINT: same trick as sigmoid_deriv.
    raise NotImplementedError("# TODO: Implement this")


def gelu(z):
    # TODO: Tanh approximation 0.5x(1+tanh(c(x+0.044715x^3))), c=sqrt(2/pi).
    # HINT: get relu+sigmoid+tanh passing first; come back to this last.
    raise NotImplementedError("# TODO: Implement this")


def gelu_deriv(z):
    # TODO: Chain rule through the approximation. Let u = c(x + 0.044715x^3):
    #   d/dx [0.5x(1+tanh(u))] = 0.5(1+tanh(u)) + 0.5x(1-tanh^2(u)) du/dx.
    # The test checks you against finite differences, so algebra errors show
    # up immediately — trust it over your own re-derivation.
    raise NotImplementedError("# TODO: Implement this")


class Perceptron:
    """Single-layer perceptron for binary labels in {0, 1}."""

    def __init__(self, n_features: int, lr: float = 0.1, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.w = rng.normal(0, 0.01, size=n_features)
        self.b = 0.0
        self.lr = float(lr)

    def predict(self, X: np.ndarray) -> np.ndarray:
        # TODO: Threshold the linear score at 0. One line.
        raise NotImplementedError("# TODO: Implement this")

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 100) -> "Perceptron":
        # TODO: Rosenblatt rule. Per epoch: err = y - predict(X) (in {-1,0,1});
        #   w += lr * (err @ X); b += lr * sum(err). Break early if no errors.
        # HINT: err @ X sums mistake vectors with their signs automatically.
        raise NotImplementedError("# TODO: Implement this")


class MLP:
    """Fully-connected network; hidden activations + linear output, MSE loss.

    Build it in this order (each step is independently testable):
      Step 1 (__init__): He-scale hidden Ws (N(0, sqrt(2/fan_in))), fan-in
        scale for the output layer, zero biases. Wrong scale = dead or
        exploding activations; print A.std() if training stalls.
      Step 2 (forward): loop layers, cache (Z, A_prev) per layer, apply the
        hidden activation from ACTIVATIONS, keep the last layer linear.
      Step 3 (fit, output grad): dA = 2(out - y) / out.size (mean-MSE grad).
      Step 4 (fit, backward loop): from last layer to first —
        dZ = dA (output) or dA * act_deriv(Z) (hidden);
        dW = A_prev.T @ dZ;  dB = sum(dZ, axis=0);  dA = dZ @ W.T;
        then W -= lr*dW, B -= lr*dB.
      Step 5: run the suite; if loss rises, halve lr twice before suspecting
        the math (guides/02_debugging.md, Story 4).
    """

    def __init__(self, layer_dims: list[int], activations="relu",
                 lr: float = 0.1, seed: int = 0):
        # TODO (Step 1): see class docstring. Accept a single string or one
        # activation per hidden layer; store self.W / self.B / self.names.
        raise NotImplementedError("# TODO: Implement this")

    def forward(self, X: np.ndarray):
        # TODO (Step 2): return (output, caches), caches = [(Z, A_prev)].
        raise NotImplementedError("# TODO: Implement this")

    def predict(self, X: np.ndarray) -> np.ndarray:
        # TODO: One line via forward (ignore the caches).
        raise NotImplementedError("# TODO: Implement this")

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 100) -> list[float]:
        # TODO (Steps 3-4): per epoch, record mean-MSE BEFORE the update;
        # return the loss history (history[-1] < history[0] is the test).
        raise NotImplementedError("# TODO: Implement this")


# Registry of YOUR functions above (scaffolding, not the answer): referencing
# them here does not call them, so this dict exists before you implement.
ACTIVATIONS = {
    "relu": (relu, relu_deriv),
    "gelu": (gelu, gelu_deriv),
    "sigmoid": (sigmoid, sigmoid_deriv),
    "tanh": (tanh, tanh_deriv),
}
