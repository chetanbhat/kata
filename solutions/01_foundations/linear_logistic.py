"""Vectorized Linear & Logistic Regression in pure NumPy.

Math objectives:
  - OLS: closed-form ridge solution  w = (X'X + l2*I)^{-1} X'y (bias unpenalized).
  - Logistic regression: full-batch gradient descent on the L1/L2-penalized
    binary cross-entropy loss, fully vectorized (no Python loops over samples).
"""

import numpy as np


def _add_bias(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    return np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)


def stable_sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid (no overflow for large |z|)."""
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    exp_z = np.exp(z[~pos])
    out[~pos] = exp_z / (1.0 + exp_z)
    return out


class LinearRegression:
    """Ordinary Least Squares with optional L2 (ridge) penalty on weights."""

    def __init__(self, l2: float = 0.0):
        self.l2 = float(l2)
        self.w: np.ndarray | None = None  # (n_features,)
        self.b: float | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegression":
        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float).reshape(-1)
        n, d = X.shape
        Xb = _add_bias(X)  # (n, d+1); bias column excluded from penalty
        penalty = self.l2 * np.eye(d + 1)
        penalty[-1, -1] = 0.0
        theta = np.linalg.solve(Xb.T @ Xb + penalty, Xb.T @ y)
        self.w, self.b = theta[:-1], float(theta[-1])
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return X @ self.w + self.b

    def mse(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean((self.predict(X) - np.asarray(y, dtype=float)) ** 2))


class LogisticRegression:
    """Binary logistic regression trained with vectorized full-batch GD.

    Loss = mean BCE + l2 * ||w||^2 + l1 * ||w||_1 (bias unpenalized).
    """

    def __init__(self, lr: float = 0.5, n_iters: int = 1000,
                 l1: float = 0.0, l2: float = 0.0):
        self.lr, self.n_iters = float(lr), int(n_iters)
        self.l1, self.l2 = float(l1), float(l2)
        self.w: np.ndarray | None = None
        self.b: float = 0.0
        self.loss_history: list[float] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegression":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        n, d = X.shape
        self.w = np.zeros(d)
        self.b = 0.0
        self.loss_history = []
        for _ in range(self.n_iters):
            p = stable_sigmoid(X @ self.w + self.b)
            self.loss_history.append(float(self._loss(p, y)))
            err = (p - y) / n  # (n,)
            grad_w = X.T @ err + 2.0 * self.l2 * self.w + self.l1 * np.sign(self.w)
            grad_b = float(np.sum(err))
            self.w -= self.lr * grad_w
            self.b -= self.lr * grad_b
        return self

    def _loss(self, p: np.ndarray, y: np.ndarray) -> float:
        eps = 1e-12
        pc = np.clip(p, eps, 1.0 - eps)
        bce = -np.mean(y * np.log(pc) + (1.0 - y) * np.log(1.0 - pc))
        return bce + self.l2 * float(self.w @ self.w) + self.l1 * float(np.sum(np.abs(self.w)))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return stable_sigmoid(np.asarray(X, dtype=float) @ self.w + self.b)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)
