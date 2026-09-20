"""Capstone 09 — your first end-to-end run (the missing experience).

Unlike Modules 1–8 (one formula at a time), here you do the whole job:
generate data -> stratified-ish shuffle split -> fit preprocessing on TRAIN
only (leakage lesson) -> train with validation monitoring -> diagnose the
curves -> early-stop + checkpoint the best -> evaluate ONCE on test.

Prereqs: guides/00, guides/01 §4 (bias–variance), Module 2 layers.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


def make_moons(n: int = 400, noise: float = 0.1, seed: int = 0):
    """Two interleaving half-circles. Returns X (n,2) float32, y (n,) int64."""
    rng = np.random.default_rng(seed)
    n_out, n_in = n // 2, n - n // 2
    t = np.linspace(0, np.pi, n_out)
    outer = np.stack([np.cos(t), np.sin(t)], axis=1)
    t = np.linspace(0, np.pi, n_in)
    inner = np.stack([1.0 - np.cos(t), 1.0 - np.sin(t) - 0.5], axis=1)
    X = np.concatenate([outer, inner], axis=0) + rng.normal(0, noise, size=(n, 2))
    y = np.concatenate([np.zeros(n_out), np.ones(n_in)]).astype(np.int64)
    return X.astype(np.float32), y


def train_val_test_split(X, y, train_frac=0.7, val_frac=0.15, seed=0):
    """Shuffled split. Returns (splits, idx): splits[name] = (X, y),
    idx[name] = row indices (disjoint, covering all rows)."""
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    perm = rng.permutation(n)
    n_train, n_val = int(n * train_frac), int(n * val_frac)
    bounds = {"train": (0, n_train), "val": (n_train, n_train + n_val),
              "test": (n_train + n_val, n)}
    splits, idx = {}, {}
    for name, (a, b) in bounds.items():
        idx[name] = perm[a:b]
        splits[name] = (X[idx[name]], y[idx[name]])
    return splits, idx


class StandardScaler:
    """Fit mean/std on TRAIN ONLY; reusing them everywhere else is the point."""

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0)
        self.scale_[self.scale_ == 0] = 1.0  # constant-column guard
        return self

    def transform(self, X):
        out = (np.asarray(X, dtype=float) - self.mean_) / self.scale_
        return out.astype(np.float32)  # keep the pipeline in float32

    def fit_transform(self, X):
        return self.fit(X).transform(X)


def make_mlp(hidden: int = 32, dropout: float = 0.0, seed: int = 0) -> nn.Module:
    torch.manual_seed(seed)
    return nn.Sequential(
        nn.Linear(2, hidden), nn.ReLU(), nn.Dropout(dropout),
        nn.Linear(hidden, 1),  # single logit; use BCEWithLogitsLoss
    )


def _acc(logits: torch.Tensor, y: torch.Tensor) -> float:
    return float(((logits.squeeze(1) >= 0).long() == y).float().mean())


def train_model(model, Xtr, ytr, Xva, yva, epochs=200, lr=1e-2, patience=20,
                seed=0, ckpt_path=None) -> dict:
    """Full-batch Adam + early stopping on val loss (restore + save best)."""
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    Xtr = torch.as_tensor(np.asarray(Xtr, dtype=np.float32))
    ytr = torch.as_tensor(np.asarray(ytr, dtype=np.int64))
    Xva = torch.as_tensor(np.asarray(Xva, dtype=np.float32))
    yva = torch.as_tensor(np.asarray(yva, dtype=np.int64))
    hist = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best, best_state, bad = float("inf"), None, 0
    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        loss = loss_fn(model(Xtr).squeeze(1), ytr.float())
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            tl, vl = float(loss), float(loss_fn(model(Xva).squeeze(1), yva.float()))
            ta, va = _acc(model(Xtr), ytr), _acc(model(Xva), yva)
        for k, v in (("train_loss", tl), ("val_loss", vl),
                     ("train_acc", ta), ("val_acc", va)):
            hist[k].append(v)
        if vl < best - 1e-9:
            best, best_state, bad = vl, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    hist["best_epoch"], hist["epochs_run"] = int(np.argmin(hist["val_loss"])), len(hist["val_loss"])
    model.load_state_dict(best_state)
    if ckpt_path is not None:
        torch.save(best_state, ckpt_path)
    return hist


def plot_curves(history: dict, path: str):
    """Two-panel PNG: losses (log-y) + accuracies."""
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(history["train_loss"], label="train")
    ax[0].plot(history["val_loss"], label="val")
    ax[0].set_yscale("log")
    ax[0].set_title("loss")
    ax[0].legend()
    ax[1].plot(history["train_acc"], label="train")
    ax[1].plot(history["val_acc"], label="val")
    ax[1].set_title("accuracy")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def run_experiment(hidden_sizes=(8, 32, 128), n=400, noise=0.12, seed=0,
                   out_prefix="/tmp/kata_capstone"):
    """Train one MLP per width, pick by val acc, evaluate test ONCE."""
    X, y = make_moons(n=n, noise=noise, seed=seed)
    splits, _ = train_val_test_split(X, y, seed=seed)
    scaler = StandardScaler().fit(splits["train"][0])
    std = {k: (scaler.transform(Xk), yk) for k, (Xk, yk) in splits.items()}
    results, best = {}, None
    for h in hidden_sizes:
        model = make_mlp(hidden=h, seed=seed)
        ckpt = f"{out_prefix}_h{h}.pt"
        hist = train_model(model, *std["train"], *std["val"],
                           epochs=200, patience=20, seed=seed, ckpt_path=ckpt)
        plot_curves(hist, f"{out_prefix}_h{h}.png")
        model.eval()
        with torch.no_grad():
            Xt = torch.as_tensor(std["test"][0])
            yt = torch.as_tensor(std["test"][1])
            test_acc = _acc(model(Xt), yt)
        results[h] = {"val_acc": max(hist["val_acc"]), "test_acc": test_acc,
                      "epochs_run": hist["epochs_run"]}
        if best is None or results[h]["val_acc"] > results[best]["val_acc"]:
            best = h
    return {"best_hidden": best, "test_acc": results[best]["test_acc"], "all": results}
