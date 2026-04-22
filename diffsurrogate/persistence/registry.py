"""Model registry.

A single source of truth mapping string names (as used in ``config.toml``
under ``[models] enabled``) to their concrete ``SurrogateModel`` class and
matching config dataclass. Used by both the CLI (to instantiate from
config) and the loader (to rebuild the right class from ``metadata.json``).

Lazy imports throughout, so a user with only ``torch`` installed doesn't
crash just because ``chaospy`` isn't importable.
"""

from __future__ import annotations

from typing import Any, Callable

from diffsurrogate.models.base import SurrogateModel


def _nn():
    from diffsurrogate.models.neural_net import NeuralNetSurrogate
    from diffsurrogate.config import NeuralNetConfig
    return NeuralNetSurrogate, NeuralNetConfig


def _gp():
    from diffsurrogate.models.gaussian_process import GaussianProcessSurrogate
    from diffsurrogate.config import GaussianProcessConfig
    return GaussianProcessSurrogate, GaussianProcessConfig


def _pinn():
    from diffsurrogate.models.pinn import PINNSurrogate
    from diffsurrogate.config import PINNConfig
    return PINNSurrogate, PINNConfig


def _fno():
    from diffsurrogate.models.fno import FNOSurrogate
    from diffsurrogate.config import FNOConfig
    return FNOSurrogate, FNOConfig


def _dgp():
    from diffsurrogate.models.deep_gp import DeepGPSurrogate
    from diffsurrogate.config import DeepGPConfig
    return DeepGPSurrogate, DeepGPConfig


def _pce():
    from diffsurrogate.models.pce import PCESurrogate
    from diffsurrogate.config import PCEConfig
    return PCESurrogate, PCEConfig


# Public: name → thunk returning (ModelClass, ConfigDataclass).
MODEL_REGISTRY: dict[str, Callable[[], tuple[type[SurrogateModel], type[Any]]]] = {
    "neural_net": _nn,
    "gaussian_process": _gp,
    "pinn": _pinn,
    "fno": _fno,
    "deep_gp": _dgp,
    "pce": _pce,
}


def class_for(name: str) -> tuple[type[SurrogateModel], type[Any]]:
    """Return ``(ModelClass, ConfigDataclass)`` for a given registered name."""
    if name not in MODEL_REGISTRY:
        raise KeyError(
            f"Unknown model name '{name}'. "
            f"Registered: {sorted(MODEL_REGISTRY)}"
        )
    return MODEL_REGISTRY[name]()
