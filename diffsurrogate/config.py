"""Configuration loader + validator.

All tunables live in ``config.toml``. This module parses that file with the
stdlib ``tomllib`` (Python 3.11+), validates it against a set of
``dataclasses``, and exposes a single ``Config`` object that the rest of the
package consumes.

No value used downstream should be hardcoded in Python — if you find a magic
number, add it here and wire it through.
"""

from __future__ import annotations

import logging
import os
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover - Python <3.11 fallback
    import tomli as tomllib  # type: ignore

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Dataclass schemas
# -----------------------------------------------------------------------------

@dataclass
class DataConfig:
    input_path: str
    predict_path: str
    input_columns: list[str]
    target_column: str
    train_fraction: float
    random_seed: int
    q2_is_prelogged: bool = True


@dataclass
class TransformsConfig:
    q2_transform: str
    w2_transform: str
    t_transform: str
    t_epsilon_frac: float
    target_scaler: str


@dataclass
class NeuralNetConfig:
    hidden_layers: list[int]
    activation: str
    use_resnet: bool
    lr: float
    weight_decay: float
    max_epochs: int
    batch_size: int = 256
    early_stopping_patience: int = 20


@dataclass
class GaussianProcessConfig:
    kernel: str
    use_white_noise: bool
    n_restarts: int
    sparse: bool
    sparse_threshold: int = 10000
    n_inducing: int = 500


@dataclass
class PINNConfig:
    hidden_layers: list[int]
    activation: str
    lr: float
    max_epochs: int
    n_collocation: int
    lambda_pde: float
    lambda_boundary: float
    alpha_s: float
    batch_size: int = 256


@dataclass
class FNOConfig:
    n_modes: int
    width: int
    n_layers: int
    lr: float
    max_epochs: int
    batch_size: int = 32


@dataclass
class DeepGPConfig:
    n_layers: int
    n_inducing: int
    n_samples: int
    lr: float
    max_epochs: int
    batch_size: int = 256


@dataclass
class PCEConfig:
    order: int
    regression: str


@dataclass
class ModelsConfig:
    enabled: list[str]
    neural_net: NeuralNetConfig | None = None
    gaussian_process: GaussianProcessConfig | None = None
    pinn: PINNConfig | None = None
    fno: FNOConfig | None = None
    deep_gp: DeepGPConfig | None = None
    pce: PCEConfig | None = None

    def get(self, name: str) -> Any:
        """Return the sub-config dataclass for a given model name."""
        return getattr(self, name, None)


@dataclass
class EvaluationConfig:
    metrics: list[str]
    output_dir: str
    save_plots: bool
    save_predictions: bool = True
    save_split_data: bool = True
    save_run_manifest: bool = True
    write_markdown_report: bool = True
    benchmark_runs_dirname: str = "benchmark_runs"


@dataclass
class PersistenceConfig:
    model_dir: str
    save_scalers: bool


@dataclass
class LoggingConfig:
    level: str = "INFO"
    log_file: str = ""


@dataclass
class Config:
    data: DataConfig
    transforms: TransformsConfig
    models: ModelsConfig
    evaluation: EvaluationConfig
    persistence: PersistenceConfig
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    # Raw source dict is kept so models can re-serialize their exact config.
    raw: dict[str, Any] = field(default_factory=dict)


# -----------------------------------------------------------------------------
# Parsing helpers
# -----------------------------------------------------------------------------

_VALID_TRANSFORMS_XY = {"log_zscore", "zscore", "log_stabilized"}
_VALID_TARGET_SCALERS = {"robust", "standard"}
_VALID_ACTIVATIONS = {"tanh", "relu", "gelu", "swish", "silu"}
_VALID_KERNELS = {"matern52_ard", "matern32_ard", "rbf_ard"}
_KNOWN_MODELS = {"neural_net", "gaussian_process", "pinn", "fno", "deep_gp", "pce"}
_VALID_METRICS = {"rmse", "nll", "coverage_95", "constraint_violation_rate", "mae"}


def _require(d: dict, key: str, section: str) -> Any:
    if key not in d:
        raise ValueError(f"[{section}] missing required key '{key}'")
    return d[key]


def _build_data(d: dict) -> DataConfig:
    cfg = DataConfig(
        input_path=_require(d, "input_path", "data"),
        predict_path=d.get("predict_path", ""),
        input_columns=list(_require(d, "input_columns", "data")),
        target_column=_require(d, "target_column", "data"),
        train_fraction=float(d.get("train_fraction", 0.8)),
        random_seed=int(d.get("random_seed", 42)),
        q2_is_prelogged=bool(d.get("q2_is_prelogged", True)),
    )
    if len(cfg.input_columns) != 3:
        raise ValueError(
            f"[data] input_columns must have exactly 3 entries "
            f"(Q2, W2, t); got {len(cfg.input_columns)}"
        )
    if not 0.0 < cfg.train_fraction < 1.0:
        raise ValueError(f"[data] train_fraction must be in (0, 1), got {cfg.train_fraction}")
    return cfg


def _build_transforms(d: dict) -> TransformsConfig:
    cfg = TransformsConfig(
        q2_transform=d.get("q2_transform", "log_zscore"),
        w2_transform=d.get("w2_transform", "log_zscore"),
        t_transform=d.get("t_transform", "log_stabilized"),
        t_epsilon_frac=float(d.get("t_epsilon_frac", 0.01)),
        target_scaler=d.get("target_scaler", "robust"),
    )
    for name, val in (("q2", cfg.q2_transform), ("w2", cfg.w2_transform), ("t", cfg.t_transform)):
        if val not in _VALID_TRANSFORMS_XY:
            raise ValueError(f"[transforms] {name}_transform='{val}' not in {_VALID_TRANSFORMS_XY}")
    if cfg.target_scaler not in _VALID_TARGET_SCALERS:
        raise ValueError(f"[transforms] target_scaler='{cfg.target_scaler}' not in {_VALID_TARGET_SCALERS}")
    return cfg


def _build_models(d: dict) -> ModelsConfig:
    enabled = list(d.get("enabled", []))
    unknown = set(enabled) - _KNOWN_MODELS
    if unknown:
        raise ValueError(f"[models] unknown models in 'enabled': {unknown}")

    def _opt(name: str, cls):
        sub = d.get(name)
        if sub is None:
            return None
        try:
            return cls(**sub)
        except TypeError as e:
            raise ValueError(f"[models.{name}] invalid config: {e}") from e

    cfg = ModelsConfig(
        enabled=enabled,
        neural_net=_opt("neural_net", NeuralNetConfig),
        gaussian_process=_opt("gaussian_process", GaussianProcessConfig),
        pinn=_opt("pinn", PINNConfig),
        fno=_opt("fno", FNOConfig),
        deep_gp=_opt("deep_gp", DeepGPConfig),
        pce=_opt("pce", PCEConfig),
    )
    # Activation & kernel sanity checks on the ones we built.
    if cfg.neural_net and cfg.neural_net.activation not in _VALID_ACTIVATIONS:
        raise ValueError(f"[models.neural_net] activation not in {_VALID_ACTIVATIONS}")
    if cfg.pinn and cfg.pinn.activation not in _VALID_ACTIVATIONS:
        raise ValueError(f"[models.pinn] activation not in {_VALID_ACTIVATIONS}")
    if cfg.gaussian_process and cfg.gaussian_process.kernel not in _VALID_KERNELS:
        raise ValueError(f"[models.gaussian_process] kernel not in {_VALID_KERNELS}")
    return cfg


def _build_evaluation(d: dict) -> EvaluationConfig:
    metrics = list(d.get("metrics", ["rmse"]))
    unknown = set(metrics) - _VALID_METRICS
    if unknown:
        raise ValueError(f"[evaluation] unknown metrics: {unknown}")
    return EvaluationConfig(
        metrics=metrics,
        output_dir=d.get("output_dir", "results/"),
        save_plots=bool(d.get("save_plots", True)),
        save_predictions=bool(d.get("save_predictions", True)),
        save_split_data=bool(d.get("save_split_data", True)),
        save_run_manifest=bool(d.get("save_run_manifest", True)),
        write_markdown_report=bool(d.get("write_markdown_report", True)),
        benchmark_runs_dirname=str(d.get("benchmark_runs_dirname", "benchmark_runs")),
    )


def _build_persistence(d: dict) -> PersistenceConfig:
    return PersistenceConfig(
        model_dir=d.get("model_dir", "saved_models/"),
        save_scalers=bool(d.get("save_scalers", True)),
    )


def _build_logging(d: dict) -> LoggingConfig:
    return LoggingConfig(
        level=d.get("level", "INFO"),
        log_file=d.get("log_file", ""),
    )


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def load_config(path: str | Path) -> Config:
    """Parse, validate, and return a Config from a TOML file.

    Also configures stdlib logging and sets all known global random seeds from
    ``[data] random_seed``. Safe to call multiple times in a process (last call
    wins for seeds and logging config).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("rb") as f:
        raw = tomllib.load(f)

    for section in ("data", "transforms", "models", "evaluation", "persistence"):
        if section not in raw:
            raise ValueError(f"Config file missing required [{section}] section")

    cfg = Config(
        data=_build_data(raw["data"]),
        transforms=_build_transforms(raw["transforms"]),
        models=_build_models(raw["models"]),
        evaluation=_build_evaluation(raw["evaluation"]),
        persistence=_build_persistence(raw["persistence"]),
        logging=_build_logging(raw.get("logging", {})),
        raw=raw,
    )
    _configure_logging(cfg.logging)
    set_global_seeds(cfg.data.random_seed)
    logger.info("Loaded config from %s", path)
    return cfg


def _configure_logging(cfg: LoggingConfig) -> None:
    level = getattr(logging, cfg.level.upper(), logging.INFO)
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if cfg.log_file:
        log_path = Path(cfg.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path))
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )


def set_global_seeds(seed: int) -> None:
    """Set seeds for numpy, python random, PYTHONHASHSEED, and (if present) torch."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch  # noqa: WPS433 (runtime optional)

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
