"""EXERCISE 1.2 — Perceptron & MLP with manual gradients (NumPy only).

Goals:
  - Perceptron: Rosenblatt mistake-driven updates, vectorized over mistakes.
  - MLP: forward caching + backprop with ReLU/GeLU/Sigmoid/Tanh.

Run: pytest tests/test_01_foundations.py -k "perceptron or mlp or activation"
"""

import numpy as np


def relu(z):
    # TODO: Implement this.
    raise NotImplementedError("# TODO: Implement this")


def relu_deriv(z):
    # TODO: Implement this.
    raise NotImplementedError("# TODO: Implement this")


def gelu(z):
    # TODO: Tanh approximation 0.5x(1+tanh(c(x+0.044715x^3))), c=sqrt(2/pi).
    raise NotImplementedError("# TODO: Implement this")


def gelu_deriv(z):
    # TODO: Implement this (chain rule through the tanh approximation).
    raise NotImplementedError("# TODO: Implement this")


class Perceptron:
    """Single-layer perceptron for binary labels in {0, 1}."""

    def __init__(self, n_features: int, lr: float = 0.1, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.w = rng.normal(0, 0.01, size=n_features)
        self.b = 0.0
        self.lr = float(lr)

    def predict(self, X: np.ndarray) -> np.ndarray:
        # TODO: Implement this (threshold the linear score at 0).
        raise NotImplementedError("# TODO: Implement this")

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 100) -> "Perceptron":
        # TODO: Apply the Rosenblatt update on misclassified points; early-stop.
        raise NotImplementedError("# TODO: Implement this")


class MLP:
    """Fully-connected network; hidden activations + linear output, MSE loss."""

    def __init__(self, layer_dims: list[int], activations="relu",
                 lr: float = 0.1, seed: int = 0):
        # TODO: He-scale hidden layers, fan-in scale for output; store params.
        raise NotImplementedError("# TODO: Implement this")

    def forward(self, X: np.ndarray):
        # TODO: Return (output, caches) with one (Z, A_prev) cache per layer.
        raise NotImplementedError("# TODO: Implement this")

    def predict(self, X: np.ndarray) -> np.ndarray:
        # TODO: Implement this.
        raise NotImplementedError("# TODO: Implement this")

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 100) -> list[float]:
        # TODO: Full-batch GD with manual backprop; return per-epoch MSE.
        raise NotImplementedError("# TODO: Implement this")
