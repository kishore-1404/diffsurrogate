"""Surrogate model implementations.

Each concrete model subclasses ``SurrogateModel`` from ``base.py``. Imports
are lazy: optional heavy dependencies (torch, gpytorch, neuraloperator,
chaospy) are imported only when the corresponding class is actually
instantiated, so the package remains usable with any subset installed.
"""

from diffsurrogate.models.base import SurrogateModel

__all__ = ["SurrogateModel", "build_model"]


def build_model(name: str, cfg):
    """Factory: instantiate a model by string name with its config sub-table."""
    # Local imports to avoid pulling torch / gpytorch etc. at package import.
    if name == "neural_net":
        from diffsurrogate.models.neural_net import NeuralNetSurrogate

        return NeuralNetSurrogate(cfg)
    if name == "gaussian_process":
        from diffsurrogate.models.gaussian_process import GaussianProcessSurrogate

        return GaussianProcessSurrogate(cfg)
    if name == "pinn":
        from diffsurrogate.models.pinn import PINNSurrogate

        return PINNSurrogate(cfg)
    if name == "fno":
        from diffsurrogate.models.fno import FNOSurrogate

        return FNOSurrogate(cfg)
    if name == "deep_gp":
        from diffsurrogate.models.deep_gp import DeepGPSurrogate

        return DeepGPSurrogate(cfg)
    if name == "pce":
        from diffsurrogate.models.pce import PCESurrogate

        return PCESurrogate(cfg)
    raise ValueError(f"Unknown model name '{name}'")
