"""Per-model smoke tests.

For each surrogate: fit on 100 points, predict on 20, verify shape/dtype and
that ``supports_uq()`` agrees with whether ``predict`` returns a non-None std.

Tests are parametrized over model name and will skip (rather than fail) if
the optional dependency for a given model isn't installed.
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest

# Map model name → dependencies that must be importable.
_DEPENDENCIES = {
    "neural_net": ["torch"],
    "gaussian_process": [],  # sklearn is a core dep
    "pinn": ["torch"],
    "fno": ["torch"],
    "deep_gp": ["torch", "gpytorch"],
    "pce": ["chaospy"],
}


def _skip_if_missing(model_name: str) -> None:
    for dep in _DEPENDENCIES[model_name]:
        try:
            importlib.import_module(dep)
        except ImportError:
            pytest.skip(f"{model_name} requires '{dep}' which is not installed")


def _default_cfg(model_name: str):
    """Minimal config dataclass for each model, keeping fits fast."""
    from diffsurrogate.config import (
        DeepGPConfig,
        FNOConfig,
        GaussianProcessConfig,
        NeuralNetConfig,
        PCEConfig,
        PINNConfig,
    )

    if model_name == "neural_net":
        return NeuralNetConfig(
            hidden_layers=[8, 8], activation="tanh", use_resnet=True,
            lr=1e-2, weight_decay=1e-5, max_epochs=5, batch_size=32,
            early_stopping_patience=3,
        )
    if model_name == "gaussian_process":
        return GaussianProcessConfig(
            kernel="matern52_ard", use_white_noise=True, n_restarts=1,
            sparse=False, sparse_threshold=10000, n_inducing=30,
        )
    if model_name == "pinn":
        return PINNConfig(
            hidden_layers=[16, 16], activation="tanh", lr=1e-2,
            max_epochs=5, n_collocation=50, lambda_pde=0.1,
            lambda_boundary=0.1, alpha_s=0.2, batch_size=32,
        )
    if model_name == "fno":
        return FNOConfig(n_modes=4, width=8, n_layers=2, lr=1e-2,
                         max_epochs=5, batch_size=8)
    if model_name == "deep_gp":
        return DeepGPConfig(n_layers=2, n_inducing=10, n_samples=3,
                            lr=1e-2, max_epochs=3, batch_size=64)
    if model_name == "pce":
        return PCEConfig(order=2, regression="lars")
    raise ValueError(model_name)


# Supports-UQ ground truth for each model.
_UQ_TRUTH = {
    "neural_net": False,
    "gaussian_process": True,
    "pinn": False,
    "fno": False,
    "deep_gp": True,
    "pce": True,
}


@pytest.mark.parametrize("model_name", list(_DEPENDENCIES))
class TestModelSmoke:
    def test_fit_predict_shapes(self, model_name, small_dataset):
        _skip_if_missing(model_name)
        from diffsurrogate.models import build_model

        X, y = small_dataset
        # Z-score inputs for fast, stable fitting.
        Xs = (X - X.mean(axis=0)) / X.std(axis=0)
        ys = (y.ravel() - y.mean()) / y.std()

        cfg = _default_cfg(model_name)
        model = build_model(model_name, cfg)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(Xs, ys)
        mu, std = model.predict(Xs[:20])

        assert mu.shape == (20,)
        assert np.issubdtype(mu.dtype, np.floating)
        assert np.isfinite(mu).all()

        if _UQ_TRUTH[model_name]:
            assert std is not None
            assert std.shape == (20,)
            assert np.issubdtype(std.dtype, np.floating)
            assert (std >= 0).all()
        else:
            assert std is None

    def test_supports_uq_flag(self, model_name):
        _skip_if_missing(model_name)
        from diffsurrogate.models import build_model
        model = build_model(model_name, _default_cfg(model_name))
        assert model.supports_uq() == _UQ_TRUTH[model_name]

    def test_name(self, model_name):
        _skip_if_missing(model_name)
        from diffsurrogate.models import build_model
        model = build_model(model_name, _default_cfg(model_name))
        assert model.name() == model_name


class TestFNOGriddedMode:
    """FNO has two code paths: fno1d (gridded) and mlp_fallback. Hit both."""

    def test_fno_detects_gridded(self):
        _skip_if_missing("fno")
        from diffsurrogate.models.fno import FNOSurrogate

        t_grid = np.linspace(-1.0, -0.05, 8)
        q_vals = [1.0, 2.0]
        w_vals = [10.0, 20.0, 40.0]
        X = np.array([[q, w, t] for q in q_vals for w in w_vals for t in t_grid])
        y = np.sin(X[:, 2]) * np.exp(-np.abs(X[:, 2]))

        cfg = _default_cfg("fno")
        m = FNOSurrogate(cfg)
        m.fit(X, y)
        assert m.mode == "fno1d"

    def test_fno_falls_back_when_not_gridded(self, small_dataset):
        _skip_if_missing("fno")
        from diffsurrogate.models.fno import FNOSurrogate

        X, y = small_dataset  # random-sampled, not gridded
        cfg = _default_cfg("fno")
        m = FNOSurrogate(cfg)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.fit(X, y.ravel())
        assert m.mode == "mlp_fallback"
