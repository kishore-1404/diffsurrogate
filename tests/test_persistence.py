"""Round-trip save/load tests for every surrogate model.

For each model: fit, predict, save (model + scalers + metadata), load into a
fresh instance, predict again, and assert predictions match. DeepGP uses a
looser tolerance because its ``predict`` draws Monte Carlo samples through
the variational stack — identical weights, non-identical forward passes.
"""

from __future__ import annotations

import importlib
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

_DEPENDENCIES = {
    "neural_net": ["torch"],
    "gaussian_process": [],
    "pinn": ["torch"],
    "fno": ["torch"],
    "deep_gp": ["torch", "gpytorch"],
    "pce": ["chaospy"],
}

# Predict tolerance per model (max abs diff in scaled-space mean).
_TOLERANCE = {
    "neural_net": 1e-6,
    "gaussian_process": 1e-6,
    "pinn": 1e-6,
    "fno": 1e-6,
    "deep_gp": 0.2,   # MC sampling — ~2.5e-2 on small data, looser for safety
    "pce": 1e-6,
}


def _skip_if_missing(model_name: str) -> None:
    for dep in _DEPENDENCIES[model_name]:
        try:
            importlib.import_module(dep)
        except ImportError:
            pytest.skip(f"{model_name} requires '{dep}' which is not installed")


def _default_cfg(model_name: str):
    """Reuse the fast-fit configs from test_models."""
    from tests.test_models import _default_cfg as _cfg
    return _cfg(model_name)


@pytest.mark.parametrize("model_name", list(_DEPENDENCIES))
class TestRoundTrip:
    def test_round_trip_via_model_save_load(self, model_name, small_dataset):
        """Direct model.save → model.load (no persistence wrapper)."""
        _skip_if_missing(model_name)
        from diffsurrogate.models import build_model

        X, y = small_dataset
        Xs = (X - X.mean(axis=0)) / X.std(axis=0)
        ys = (y.ravel() - y.mean()) / y.std()

        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            m1 = build_model(model_name, _default_cfg(model_name))
            m1.fit(Xs, ys)
            mu1, _ = m1.predict(Xs[:10])

            with tempfile.TemporaryDirectory() as d:
                d = Path(d) / "model"
                m1.save(d)
                assert d.exists() and any(d.iterdir())

                m2 = build_model(model_name, _default_cfg(model_name))
                m2.load(d)
                mu2, _ = m2.predict(Xs[:10])

        diff = float(np.max(np.abs(mu1 - mu2)))
        tol = _TOLERANCE[model_name]
        assert diff < tol, f"{model_name} round-trip diff {diff} > tol {tol}"

    def test_round_trip_via_persistence(self, model_name, small_dataset):
        """Full persistence layer: save_model + load_model."""
        _skip_if_missing(model_name)
        from diffsurrogate.models import build_model
        from diffsurrogate.persistence import load_model, save_model

        X, y = small_dataset
        Xs = (X - X.mean(axis=0)) / X.std(axis=0)
        ys = (y.ravel() - y.mean()) / y.std()

        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            m1 = build_model(model_name, _default_cfg(model_name))
            m1.fit(Xs, ys)
            mu1, _ = m1.predict(Xs[:10])

            with tempfile.TemporaryDirectory() as d:
                d = Path(d) / "artifact"
                save_model(
                    m1,
                    {"input_scaler": "placeholder", "target_scaler": "placeholder"},
                    d,
                    n_train=len(Xs),
                    config_snapshot={"test": True},
                )

                # metadata.json sanity check
                with (d / "metadata.json").open() as f:
                    meta = json.load(f)
                assert meta["model_name"] == model_name
                assert meta["n_train"] == len(Xs)
                assert meta["supports_uq"] == m1.supports_uq()
                assert meta["config_snapshot"] == {"test": True}

                # Load and re-predict
                m2, scalers = load_model(model_name, d, model_config=_default_cfg(model_name))
                mu2, _ = m2.predict(Xs[:10])
                assert scalers["input_scaler"] == "placeholder"
                assert scalers["target_scaler"] == "placeholder"

        diff = float(np.max(np.abs(mu1 - mu2)))
        tol = _TOLERANCE[model_name]
        assert diff < tol, f"{model_name} round-trip diff {diff} > tol {tol}"


class TestPersistenceEdgeCases:
    def test_load_nonexistent_raises(self):
        from diffsurrogate.persistence import load_model
        with pytest.raises(FileNotFoundError):
            load_model("neural_net", Path("/tmp/does-not-exist-diffsurrogate"))

    def test_load_missing_scalers_returns_empty_dict(self, small_dataset, tmp_path):
        """If scalers.joblib is absent, loader returns {} with a warning."""
        import warnings

        from diffsurrogate.models import build_model
        from diffsurrogate.persistence import load_model, save_model

        X, y = small_dataset
        Xs = (X - X.mean(axis=0)) / X.std(axis=0)
        ys = (y.ravel() - y.mean()) / y.std()
        cfg = _default_cfg("neural_net")

        m = build_model("neural_net", cfg)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.fit(Xs, ys)
        save_dir = tmp_path / "no_scalers"
        save_model(m, scalers=None, base_dir=save_dir)
        assert not (save_dir / "scalers.joblib").exists()

        m2, scalers = load_model("neural_net", save_dir, model_config=cfg)
        assert scalers == {}

    def test_atomic_save_replaces_existing(self, small_dataset, tmp_path):
        """Saving to an already-existing dir succeeds and replaces contents."""
        import warnings

        from diffsurrogate.models import build_model
        from diffsurrogate.persistence import save_model

        X, y = small_dataset
        Xs = (X - X.mean(axis=0)) / X.std(axis=0)
        ys = (y.ravel() - y.mean()) / y.std()

        m = build_model("pce", _default_cfg("pce"))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.fit(Xs, ys)
        save_dir = tmp_path / "pce_art"
        save_model(m, None, save_dir)
        assert save_dir.exists()
        # Save again — should succeed.
        save_model(m, None, save_dir)
        assert save_dir.exists()
        # No leftover .partial or .backup
        assert not save_dir.with_suffix(".partial").exists()
        assert not save_dir.with_suffix(".backup").exists()

    def test_registry_contains_all_models(self):
        from diffsurrogate.persistence.registry import MODEL_REGISTRY
        assert set(MODEL_REGISTRY) == {
            "neural_net", "gaussian_process", "pinn",
            "fno", "deep_gp", "pce",
        }

    def test_class_for_unknown_raises(self):
        from diffsurrogate.persistence.registry import class_for
        with pytest.raises(KeyError):
            class_for("nonexistent_model")
