"""Loader for saved-model directories.

Reads ``metadata.json``, picks the correct class out of ``MODEL_REGISTRY``,
and restores the model. The *current* config is used to instantiate the
model class (its hyperparameters are needed only to shape the PyTorch
module before ``load_state_dict`` — for exact reproduction, the on-disk
model-specific meta (``nn_meta.json`` etc.) overrides architecture
details).

If the class expects a config dataclass that's not available from the
current run (e.g. loading from production without a fresh config.toml in
hand), the caller can pass one via ``model_config`` or rely on the
model's own ``load()`` to pull architecture from on-disk meta.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib

from diffsurrogate.models.base import SurrogateModel
from diffsurrogate.persistence.registry import class_for

logger = logging.getLogger(__name__)


def load_model(
    model_name: str,
    base_dir: Path | str,
    model_config: Any | None = None,
) -> tuple[SurrogateModel, dict[str, Any]]:
    """Restore a saved model and its scalers.

    Parameters
    ----------
    model_name : str
        Name as used in the registry (e.g. ``"neural_net"``).
    base_dir : Path
        Directory containing ``metadata.json`` plus model-specific artifacts.
    model_config : dataclass | None
        Config to instantiate the model with. For most model classes, the
        actual architecture is re-read from on-disk meta inside ``load()``,
        so any non-None config of the right dataclass type is acceptable.
        If ``None``, we build a minimal default config so ``load()`` can run.

    Returns
    -------
    (model, scalers) : tuple
        ``model`` is a fully-restored ``SurrogateModel`` ready for
        ``predict()``. ``scalers`` is whatever dict was saved (usually
        ``{"input_scaler": ..., "target_scaler": ...}``) or ``{}``.
    """
    base_dir = Path(base_dir)
    if not base_dir.exists():
        raise FileNotFoundError(f"Saved-model directory does not exist: {base_dir}")

    meta_path = base_dir / "metadata.json"
    if meta_path.exists():
        with meta_path.open() as f:
            meta = json.load(f)
        if meta.get("model_name") and meta["model_name"] != model_name:
            logger.warning(
                "metadata.json says model_name='%s' but caller asked for '%s'",
                meta["model_name"], model_name,
            )
    else:
        logger.warning("No metadata.json in %s — proceeding without it", base_dir)

    model_cls, config_cls = class_for(model_name)
    cfg = model_config if model_config is not None else _default_config(config_cls)
    model = model_cls(cfg)
    model.load(base_dir)

    scalers_path = base_dir / "scalers.joblib"
    if scalers_path.exists():
        scalers = joblib.load(scalers_path)
    else:
        scalers = {}
        logger.warning("No scalers.joblib in %s; scalers dict is empty", base_dir)

    logger.info("Loaded %s from %s", model_name, base_dir)
    return model, scalers


def _default_config(config_cls: type) -> Any:
    """Construct a minimal instance of a config dataclass for load-time use."""
    defaults = {
        "NeuralNetConfig": dict(
            hidden_layers=[16, 16], activation="tanh", use_resnet=True,
            lr=1e-3, weight_decay=1e-5, max_epochs=1, batch_size=16,
            early_stopping_patience=1,
        ),
        "GaussianProcessConfig": dict(
            kernel="matern52_ard", use_white_noise=True, n_restarts=1,
            sparse=False, sparse_threshold=10000, n_inducing=50,
        ),
        "PINNConfig": dict(
            hidden_layers=[16, 16], activation="tanh", lr=1e-3, max_epochs=1,
            n_collocation=10, lambda_pde=1.0, lambda_boundary=1.0,
            alpha_s=0.2, batch_size=16,
        ),
        "FNOConfig": dict(n_modes=4, width=16, n_layers=2, lr=1e-3, max_epochs=1, batch_size=4),
        "DeepGPConfig": dict(n_layers=2, n_inducing=10, n_samples=3, lr=1e-2, max_epochs=1, batch_size=16),
        "PCEConfig": dict(order=2, regression="lars"),
    }
    key = config_cls.__name__
    if key not in defaults:
        raise ValueError(f"No default template for config class {key}")
    return config_cls(**defaults[key])
