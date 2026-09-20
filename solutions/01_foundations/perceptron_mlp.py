"""Perceptron and n-layer MLP with manual gradients (NumPy only).

Math objectives:
  - Perceptron: mistake-driven linear updates (Rosenblatt rule).
  - MLP: forward caching + backprop via the chain rule with customizable
    activations (ReLU, GeLU, Sigmoid, Tanh), full-batch gradient descent.
"""

import numpy as np

_TANH_COEF = np.sqrt(2.0 / np.pi)


def relu(z):
    return np.maximum(0.0, z)


def relu_deriv(z):
    return (z > 0).astype(float)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def sigmoid_deriv(z):
    s = sigmoid(z)
    return s * (1.0 - s)


def tanh(z):
    return np.tanh(z)


def tanh_deriv(z):
    return 1.0 - np.tanh(z) ** 2


def gelu(z):
    """Tanh-approximated GeLU: 0.5x(1 + tanh(c(x + 0.044715x^3)))."""
    z = np.asarray(z, dtype=float)
    return 0.5 * z * (1.0 + np.tanh(_TANH_COEF * (z + 0.044715 * z ** 3)))


def gelu_deriv(z):
    z = np.asarray(z, dtype=float)
    u = _TANH_COEF * (z + 0.044715 * z ** 3)
    t = np.tanh(u)
    du = _TANH_COEF * (1.0 + 3.0 * 0.044715 * z ** 2)
    return 0.5 * (1.0 + t) + 0.5 * z * (1.0 - t ** 2) * du


ACTIVATIONS = {
    "relu": (relu, relu_deriv),
    "gelu": (gelu, gelu_deriv),
    "sigmoid": (sigmoid, sigmoid_deriv),
    "tanh": (tanh, tanh_deriv),
}


class Perceptron:
    """Single-layer perceptron for binary labels in {0, 1}."""

    def __init__(self, n_features: int, lr: float = 0.1, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.w = rng.normal(0, 0.01, size=n_features)
        self.b = 0.0
        self.lr = float(lr)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (np.asarray(X, dtype=float) @ self.w + self.b >= 0).astype(int)

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 100) -> "Perceptron":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).reshape(-1)
        for _ in range(epochs):
            pred = self.predict(X)
            err = y - pred  # in {-1, 0, 1}
            if not np.any(err):
                break
            # Vectorized Rosenblatt update summed over mistakes.
            self.w += self.lr * (err @ X)
            self.b += self.lr * float(np.sum(err))
        return self


class MLP:
    """Fully-connected network; hidden activations + linear output, MSE loss."""

    def __init__(self, layer_dims: list[int], activations: list[str] | str = "relu",
                 lr: float = 0.1, seed: int = 0):
        if isinstance(activations, str):
            activations = [activations] * (len(layer_dims) - 2)
        assert len(activations) == len(layer_dims) - 2, "one activation per hidden layer"
        rng = np.random.default_rng(seed)
        self.lr = float(lr)
        self.names = list(activations)
        self.W: list[np.ndarray] = []
        self.B: list[np.ndarray] = []
        for i in range(len(layer_dims) - 1):
            fan_in, fan_out = layer_dims[i], layer_dims[i + 1]
            scale = np.sqrt(2.0 / fan_in) if i < len(layer_dims) - 2 else np.sqrt(1.0 / fan_in)
            self.W.append(rng.normal(0, scale, size=(fan_in, fan_out)))
            self.B.append(np.zeros(fan_out))

    def forward(self, X: np.ndarray):
        """Returns (output, caches) with caches = [(Z, A_prev)] per layer."""
        A = np.asarray(X, dtype=float)
        caches = []
        for i, (W, B) in enumerate(zip(self.W, self.B)):
            Z = A @ W + B
            caches.append((Z, A))
            if i < len(self.W) - 1:
                A = ACTIVATIONS[self.names[i]][0](Z)
            else:
                A = Z
        return A, caches

    def predict(self, X: np.ndarray) -> np.ndarray:
        out, _ = self.forward(X)
        return out

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 100) -> list[float]:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(X.shape[0], -1)
        n = X.shape[0]
        history = []
        for _ in range(epochs):
            out, caches = self.forward(X)
            loss = float(np.mean((out - y) ** 2))
            history.append(loss)
            dA = 2.0 * (out - y) / out.size
            for i in reversed(range(len(self.W))):
                Z, A_prev = caches[i]
                dZ = dA if i == len(self.W) - 1 else dA * ACTIVATIONS[self.names[i]][1](Z)
                dW = A_prev.T @ dZ
                dB = np.sum(dZ, axis=0)
                dA = dZ @ self.W[i].T
                self.W[i] -= self.lr * dW
                self.B[i] -= self.lr * dB
        return history
