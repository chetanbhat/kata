"""EXERCISE 9 — Capstone: two-moons classification end to end.

The whole job, in order (see guides/00_start_here.md + Module 09 tests):
  1. make_moons: two noisy half-circles, labels 0/1.
  2. train_val_test_split: shuffled, disjoint, covering; return index maps.
  3. StandardScaler: fit on TRAIN ONLY (leakage lesson); guard zero variance.
  4. make_mlp + train_model: full-batch Adam, val monitoring, early
     stopping with best-restore, optional checkpoint save.
  5. plot_curves + run_experiment: pick width by VAL, evaluate test ONCE.

Run: pytest tests/test_09_capstone.py -q  (your code by default)
Reading: readings/09_capstone/end_to_end.md
"""

import numpy as np
import torch
import torch.nn as nn


def make_moons(n: int = 400, noise: float = 0.1, seed: int = 0):
    # TODO: Outer arc (cos, sin), inner arc (1-cos, 1-sin-0.5) + noise.
    # Return X (n,2) float32, y (n,) int64.
    raise NotImplementedError("# TODO: Implement this")


def train_val_test_split(X, y, train_frac=0.7, val_frac=0.15, seed=0):
    # TODO: rng.permutation split; return (splits, idx) with idx disjoint.
    raise NotImplementedError("# TODO: Implement this")


class StandardScaler:
    # TODO: fit stores mean_/scale_ (1.0 where variance is 0); transform applies.
    def fit(self, X):
        raise NotImplementedError("# TODO: Implement this")

    def transform(self, X):
        raise NotImplementedError("# TODO: Implement this")

    def fit_transform(self, X):
        return self.fit(X).transform(X)


def make_mlp(hidden: int = 32, dropout: float = 0.0, seed: int = 0) -> nn.Module:
    # TODO: 2 -> hidden -> ReLU -> Dropout -> 1 logit; seed the init.
    raise NotImplementedError("# TODO: Implement this")


def train_model(model, Xtr, ytr, Xva, yva, epochs=200, lr=1e-2, patience=20,
                seed=0, ckpt_path=None) -> dict:
    # TODO: Adam + BCEWithLogits; per-epoch train/val loss+acc; early-stop on
    # val loss, restore best, save checkpoint; return history dict.
    raise NotImplementedError("# TODO: Implement this")


def plot_curves(history: dict, path: str):
    # TODO: Two-panel PNG (log-y losses, accuracies); return path.
    raise NotImplementedError("# TODO: Implement this")


def run_experiment(hidden_sizes=(8, 32, 128), n=400, noise=0.12, seed=0,
                   out_prefix="/tmp/kata_capstone"):
    # TODO: For each width: scale -> train -> checkpoint+plot -> test acc ONCE.
    # Return {"best_hidden", "test_acc", "all"} picked by val acc.
    raise NotImplementedError("# TODO: Implement this")
