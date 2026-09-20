"""Module 09 tests: splits, scaling discipline, curves, overfitting, checkpoints."""

import os

import numpy as np
import torch
import torch.nn as nn
import pytest

from tests.impl import load

ee = load("09_capstone.end_to_end")


def assert_grads_alive(module: nn.Module):
    for p in [p for p in module.parameters() if p.requires_grad]:
        assert p.grad is not None and torch.any(p.grad != 0)


def test_split_sizes_disjoint_and_covering():
    X, y = ee.make_moons(n=400, seed=0)
    assert X.shape == (400, 2) and y.shape == (400,)
    splits, idx = ee.train_val_test_split(X, y, seed=0)
    assert splits["train"][0].shape == (280, 2)
    assert splits["val"][0].shape == (60, 2)
    assert splits["test"][0].shape == (60, 2)
    all_idx = np.concatenate([idx["train"], idx["val"], idx["test"]])
    assert sorted(all_idx.tolist()) == list(range(400))  # disjoint + covering
    _, idx2 = ee.train_val_test_split(X, y, seed=0)
    assert np.array_equal(idx["train"], idx2["train"])  # deterministic


def test_scaler_fit_on_train_only_and_constant_guard():
    X, y = ee.make_moons(n=200, seed=1)
    splits, _ = ee.train_val_test_split(X, y, seed=1)
    scaler = ee.StandardScaler().fit(splits["train"][0])
    # Leakage check: stats come from train, not from the pooled data.
    assert np.allclose(scaler.mean_, splits["train"][0].mean(axis=0))
    assert not np.allclose(scaler.mean_, X.mean(axis=0))  # pooled stats would leak
    assert np.allclose(scaler.transform(splits["train"][0]).mean(axis=0), 0, atol=1e-6)
    const = np.ones((10, 2))
    assert np.all(np.isfinite(ee.StandardScaler().fit_transform(const)))


def test_training_shape_grad_convergence_and_early_stopping():
    X, y = ee.make_moons(n=300, seed=2)
    splits, _ = ee.train_val_test_split(X, y, seed=2)
    scaler = ee.StandardScaler().fit(splits["train"][0])
    std = {k: (scaler.transform(Xk), yk) for k, (Xk, yk) in splits.items()}
    model = ee.make_mlp(hidden=32, seed=0)
    Xtr = torch.as_tensor(std["train"][0])
    ytr = torch.as_tensor(std["train"][1])
    model.zero_grad()
    nn.BCEWithLogitsLoss()(model(Xtr).squeeze(1), ytr.float()).backward()
    assert_grads_alive(model)
    hist = ee.train_model(ee.make_mlp(hidden=32, seed=0), *std["train"], *std["val"],
                          epochs=200, patience=200, seed=0)
    assert len(hist["train_loss"]) == 200
    assert hist["train_loss"][-1] < hist["train_loss"][0]
    # Early stopping needs a task that actually bottoms out: small + noisy.
    Xn, yn = ee.make_moons(n=80, noise=0.25, seed=9)
    splitsn, _ = ee.train_val_test_split(Xn, yn, seed=9)
    scalern = ee.StandardScaler().fit(splitsn["train"][0])
    stdn = {k: (scalern.transform(Xk), yk) for k, (Xk, yk) in splitsn.items()}
    hist_es = ee.train_model(ee.make_mlp(hidden=64, seed=0), *stdn["train"], *stdn["val"],
                             epochs=200, patience=5, seed=0)
    assert hist_es["epochs_run"] < 200  # stopped early
    assert min(hist_es["val_loss"]) <= hist_es["val_loss"][-1] + 1e-9


def test_overfitting_gap_is_diagnosable():
    X, y = ee.make_moons(n=80, noise=0.25, seed=3)  # small + noisy
    splits, _ = ee.train_val_test_split(X, y, seed=3)
    scaler = ee.StandardScaler().fit(splits["train"][0])
    std = {k: (scaler.transform(Xk), yk) for k, (Xk, yk) in splits.items()}
    hist = ee.train_model(ee.make_mlp(hidden=128, seed=0), *std["train"], *std["val"],
                          epochs=300, patience=300, seed=0)
    assert hist["train_acc"][-1] > hist["val_acc"][-1] + 0.05  # memorized, not learned


def test_checkpoint_reload_reproduces_and_plot_exists(tmp_path):
    X, y = ee.make_moons(n=200, seed=4)
    splits, _ = ee.train_val_test_split(X, y, seed=4)
    scaler = ee.StandardScaler().fit(splits["train"][0])
    std = {k: (scaler.transform(Xk), yk) for k, (Xk, yk) in splits.items()}
    ckpt, png = str(tmp_path / "best.pt"), str(tmp_path / "curves.png")
    model = ee.make_mlp(hidden=16, seed=0)
    hist = ee.train_model(model, *std["train"], *std["val"],
                          epochs=50, patience=50, seed=0, ckpt_path=ckpt)
    assert os.path.exists(ckpt)
    fresh = ee.make_mlp(hidden=16, seed=999)  # different init
    fresh.load_state_dict(torch.load(ckpt, weights_only=True))
    fresh.eval()
    model.eval()
    with torch.no_grad():
        Xt = torch.as_tensor(std["test"][0])
        assert torch.equal((fresh(Xt).squeeze(1) >= 0).long(),
                           (model(Xt).squeeze(1) >= 0).long())
    ee.plot_curves(hist, png)
    assert os.path.exists(png) and os.path.getsize(png) > 0


def test_experiment_picks_by_val_and_tests_once(tmp_path):
    res = ee.run_experiment(hidden_sizes=(8, 32), n=300, seed=5,
                            out_prefix=str(tmp_path / "exp"))
    assert res["best_hidden"] in (8, 32)
    assert res["test_acc"] > 0.85
    assert set(res["all"]) == {8, 32}
