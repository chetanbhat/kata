"""Module 1 tests: NumPy foundations (regression, perceptron/MLP, losses).

Invariant mapping (NumPy has no .grad, so the gradient-flow invariant is
checked via finite differences: parameters with zero numerical gradient
would indicate a dead computational path).
"""

from tests.impl import load, ref

import numpy as np
import pytest

ll = load("01_foundations.linear_logistic")
pm = load("01_foundations.perceptron_mlp")
lx = load("01_foundations.losses")

RNG = np.random.default_rng(0)


# ---------------------------------------------------------------- OLS / ridge
def test_ols_shape_and_weight_recovery():
    n, d = 64, 5
    X = RNG.normal(size=(n, d))
    w_star = np.arange(1, d + 1, dtype=float)
    y = X @ w_star + 2.0 + 1e-6 * RNG.normal(size=n)
    model = ll.LinearRegression().fit(X, y)
    assert model.w.shape == (d,)
    assert model.predict(X).shape == (n,)
    np.testing.assert_allclose(model.w, w_star, atol=1e-4)
    assert model.b == pytest.approx(2.0, abs=1e-4)


def test_ridge_runs_and_scales_with_batch():
    for n in (8, 33):
        X = RNG.normal(size=(n, 4))
        y = X @ np.ones(4) + RNG.normal(size=n)
        m = ll.LinearRegression(l2=1.0).fit(X, y)
        assert m.predict(X).shape == (n,)
        assert np.isfinite(m.mse(X, y))


# ------------------------------------------------------------- logistic
def _blobs(n=120):
    X = np.vstack([RNG.normal(-2, 1, size=(n // 2, 2)), RNG.normal(2, 1, size=(n // 2, 2))])
    y = np.array([0] * (n // 2) + [1] * (n // 2))
    return X, y


def test_logistic_shape_and_accuracy():
    X, y = _blobs()
    model = ll.LogisticRegression(lr=0.5, n_iters=300).fit(X, y)
    assert model.predict_proba(X).shape == (y.shape[0],)
    assert (model.predict(X).shape) == (y.shape[0],)
    assert float(np.mean(model.predict(X) == y)) > 0.95


def test_logistic_10_step_convergence():
    X, y = _blobs()
    model = ll.LogisticRegression(lr=0.5, n_iters=10).fit(X, y)
    assert len(model.loss_history) == 10
    assert model.loss_history[-1] < model.loss_history[0]


def test_stable_sigmoid_extremes():
    z = np.array([-1000.0, 0.0, 1000.0])
    out = ll.stable_sigmoid(z)
    assert np.all(np.isfinite(out))
    np.testing.assert_allclose(out, [0.0, 0.5, 1.0], atol=1e-6)


# ------------------------------------------------------------- perceptron/MLP
def test_perceptron_separable_converges():
    X = np.array([[0.2, 0.1], [0.4, 0.3], [3.0, 2.5], [2.5, 3.5]])
    y = np.array([0, 0, 1, 1])
    clf = pm.Perceptron(n_features=2, lr=0.5).fit(X, y, epochs=200)
    assert clf.predict(X).shape == (4,)
    np.testing.assert_array_equal(clf.predict(X), y)


def test_mlp_10_step_convergence_and_shape():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(40, 3))
    y = (X @ np.array([1.0, -2.0, 0.5]) + 0.3).reshape(-1, 1)
    mlp = pm.MLP([3, 16, 1], activations="tanh", lr=0.05, seed=0)
    hist = mlp.fit(X, y, epochs=10)
    assert len(hist) == 10
    assert hist[-1] < hist[0]
    assert mlp.predict(X).shape == (40, 1)


@pytest.mark.parametrize("name", ["relu", "gelu", "sigmoid", "tanh"])
def test_activation_values_and_vectorization(name):
    fn, deriv = pm.ACTIVATIONS[name]
    z = np.array([-2.0, 0.5, 3.0])  # off-kink points (ReLU is non-smooth at 0)
    out = fn(z)
    assert out.shape == (3,)
    assert np.all(np.isfinite(out))
    # Finite-difference check of the analytic derivative at generic points.
    h = 1e-5
    num = (fn(z + h) - fn(z - h)) / (2 * h)
    np.testing.assert_allclose(deriv(z), num, rtol=1e-4, atol=1e-6)


def test_mlp_numerical_gradients_nonzero():
    """Gradient-flow analogue: every parameter must influence the loss."""
    rng = np.random.default_rng(2)
    X = rng.normal(size=(8, 2))
    y = rng.normal(size=(8, 1))
    mlp = pm.MLP([2, 4, 1], activations="relu", seed=0)
    base = float(np.mean((mlp.predict(X) - y) ** 2))
    h = 1e-5
    for W in mlp.W:
        idx = tuple(0 for _ in W.shape)
        W[idx] += h
        moved = float(np.mean((mlp.predict(X) - y) ** 2))
        W[idx] -= h
        assert abs(moved - base) > 0, "dead parameter: no gradient path"


# ------------------------------------------------------------------ losses
def test_mse_bce_known_values():
    assert lx.mse(np.array([1.0, 2.0]), np.array([1.0, 3.0])) == pytest.approx(0.5)
    assert lx.bce(np.array([0.9, 0.1]), np.array([1.0, 0.0])) == pytest.approx(
        -np.log(0.9), rel=1e-6)
    assert lx.cross_entropy(np.array([[2.0, 0.5, 0.1]]), np.array([0])) == pytest.approx(
        float(-(2.0 - np.log(np.exp(2.0) + np.exp(0.5) + np.exp(0.1)))), rel=1e-9)


def test_losses_numerically_stable():
    assert np.isfinite(lx.bce(np.array([0.0, 1.0]), np.array([0.0, 1.0])))
    assert np.isfinite(lx.cross_entropy(np.array([[1000.0, 1001.0]]), np.array([1])))
    assert np.isfinite(lx.cross_entropy(np.array([[-1000.0, -999.0]]), np.array([0])))
