"""EXERCISE 1.1 — Vectorized Linear & Logistic Regression (NumPy only).

Goals:
  - LinearRegression.fit: closed-form ridge solution, bias unpenalized.
  - LogisticRegression.fit: full-batch GD on L1/L2-penalized BCE, zero loops.
  - stable_sigmoid: no overflow for large |z|.

Run: pytest tests/test_01_foundations.py -k "regression or ols or logistic"
"""

import numpy as np


def _add_bias(X: np.ndarray) -> np.ndarray:
    # TODO: Append a ones column. Shape (n, d) -> (n, d+1).
    raise NotImplementedError("# TODO: Implement this")


def stable_sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    # TODO: Implement this (branch on z >= 0 to avoid exp overflow).
    raise NotImplementedError("# TODO: Implement this")


class LinearRegression:
    """OLS with optional L2 penalty (bias excluded from the penalty)."""

    def __init__(self, l2: float = 0.0):
        self.l2 = float(l2)
        self.w = None
        self.b = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegression":
        # TODO: Solve (Xb'Xb + P)^{-1} Xb'y with np.linalg.solve.
        raise NotImplementedError("# TODO: Implement this")

    def predict(self, X: np.ndarray) -> np.ndarray:
        # TODO: Implement this.
        raise NotImplementedError("# TODO: Implement this")

    def mse(self, X: np.ndarray, y: np.ndarray) -> float:
        # TODO: Implement this.
        raise NotImplementedError("# TODO: Implement this")


class LogisticRegression:
    """Binary logistic regression via vectorized full-batch gradient descent."""

    def __init__(self, lr: float = 0.5, n_iters: int = 1000,
                 l1: float = 0.0, l2: float = 0.0):
        self.lr, self.n_iters = float(lr), int(n_iters)
        self.l1, self.l2 = float(l1), float(l2)
        self.w = None
        self.b = 0.0
        self.loss_history: list[float] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegression":
        # TODO: Loop n_iters: sigmoid forward, BCE loss, vectorized gradients.
        raise NotImplementedError("# TODO: Implement this")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        # TODO: Implement this.
        raise NotImplementedError("# TODO: Implement this")

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        # TODO: Implement this.
        raise NotImplementedError("# TODO: Implement this")
