"""EXERCISE 1.3 — Loss functions & numerical stability (NumPy).

Goals: MSE, clipped BCE, and cross-entropy from raw logits stabilized with
the log-sum-exp trick (subtract the row max before exponentiating).

Run: pytest tests/test_01_foundations.py -k "loss or cross_entropy or bce"
"""

import numpy as np


def mse(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    # TODO: Implement this.
    raise NotImplementedError("# TODO: Implement this")


def bce(y_pred: np.ndarray, y_true: np.ndarray, eps: float = 1e-12) -> float:
    # TODO: Clip probabilities to [eps, 1-eps], then mean BCE.
    raise NotImplementedError("# TODO: Implement this")


def log_softmax(logits: np.ndarray) -> np.ndarray:
    # TODO: Row-wise log-softmax with the log-sum-exp trick.
    raise NotImplementedError("# TODO: Implement this")


def cross_entropy(logits: np.ndarray, targets: np.ndarray) -> float:
    # TODO: Mean negative log-probability of the target classes.
    raise NotImplementedError("# TODO: Implement this")
