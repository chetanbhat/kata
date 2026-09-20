"""Loss functions with numerical stabilization (NumPy, vectorized).

Math objectives:
  - MSE: mean squared error.
  - BCE: binary cross-entropy, stable via clipped probabilities.
  - Cross-entropy: multi-class loss from raw logits, stable via the
    log-sum-exp trick (subtract row max before exp).
"""

import numpy as np


def mse(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    y_pred = np.asarray(y_pred, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    return float(np.mean((y_pred - y_true) ** 2))


def bce(y_pred: np.ndarray, y_true: np.ndarray, eps: float = 1e-12) -> float:
    """Binary cross-entropy; probabilities clipped to [eps, 1-eps]."""
    p = np.clip(np.asarray(y_pred, dtype=float), eps, 1.0 - eps)
    y = np.asarray(y_true, dtype=float)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def log_softmax(logits: np.ndarray) -> np.ndarray:
    """Row-wise log-softmax using the log-sum-exp trick."""
    z = np.asarray(logits, dtype=float)
    z = z - z.max(axis=-1, keepdims=True)
    return z - np.log(np.sum(np.exp(z), axis=-1, keepdims=True))


def cross_entropy(logits: np.ndarray, targets: np.ndarray) -> float:
    """Mean cross-entropy from raw logits; targets are class indices."""
    log_p = log_softmax(logits)
    t = np.asarray(targets).reshape(-1)
    n = log_p.shape[0]
    return float(-np.mean(log_p[np.arange(n), t]))
