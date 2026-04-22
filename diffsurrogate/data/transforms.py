"""Physics-informed data transforms.

Three input scalers (one per kinematic axis) plus two target scalers. Each has
the standard ``fit`` / ``transform`` / ``inverse_transform`` API, keeps its
fitted state as plain attributes, and is joblib-picklable so it can be saved
alongside the model weights.

The ``t`` axis gets special treatment because |t| can sit right at zero where
the diffractive dips live — we stabilize with
    eps = t_epsilon_frac * median(|t|)
computed on the *training split only* (never on test data).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Input scalers — per-axis
# =============================================================================


class ZScoreScaler:
    """Plain z-score: ``(x - mean) / std``. Used when an input is already
    on an appropriate scale (e.g. Q2 is stored pre-logged)."""

    def __init__(self) -> None:
        self.mean_: float | None = None
        self.std_: float | None = None

    def fit(self, x: np.ndarray) -> "ZScoreScaler":
        x = np.asarray(x, dtype=np.float64).ravel()
        self.mean_ = float(np.mean(x))
        std = float(np.std(x))
        self.std_ = std if std > 0 else 1.0
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        self._check_fitted()
        x = np.asarray(x, dtype=np.float64)
        return (x - self.mean_) / self.std_

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        self._check_fitted()
        x = np.asarray(x, dtype=np.float64)
        return x * self.std_ + self.mean_

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return self.fit(x).transform(x)

    def _check_fitted(self) -> None:
        if self.mean_ is None:
            raise RuntimeError(f"{type(self).__name__} used before fit()")


class LogZScoreScaler:
    """``log(x)`` then z-score. For strictly-positive inputs (Q², W²)."""

    def __init__(self) -> None:
        self.mean_: float | None = None
        self.std_: float | None = None

    def fit(self, x: np.ndarray) -> "LogZScoreScaler":
        x = np.asarray(x, dtype=np.float64).ravel()
        if np.any(x <= 0):
            raise ValueError(
                "LogZScoreScaler expects strictly positive values; "
                f"got min={x.min()}"
            )
        lx = np.log(x)
        self.mean_ = float(np.mean(lx))
        std = float(np.std(lx))
        self.std_ = std if std > 0 else 1.0
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        self._check_fitted()
        x = np.asarray(x, dtype=np.float64)
        if np.any(x <= 0):
            raise ValueError("LogZScoreScaler.transform: non-positive input")
        return (np.log(x) - self.mean_) / self.std_

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        self._check_fitted()
        x = np.asarray(x, dtype=np.float64)
        return np.exp(x * self.std_ + self.mean_)

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return self.fit(x).transform(x)

    def _check_fitted(self) -> None:
        if self.mean_ is None:
            raise RuntimeError(f"{type(self).__name__} used before fit()")


class LogStabilizedScaler:
    """``log(|x| + eps)`` then z-score, with sign preserved through inverse.

    The epsilon is computed as ``t_epsilon_frac * median(|x|)`` from training
    data. Since ``t`` is typically negative (Mandelstam momentum transfer),
    the inverse transform restores the sign of the *training* sign convention:
    we record the dominant sign during fit and apply it on inverse.
    """

    def __init__(self, t_epsilon_frac: float = 0.01) -> None:
        self.t_epsilon_frac = float(t_epsilon_frac)
        self.eps_: float | None = None
        self.mean_: float | None = None
        self.std_: float | None = None
        self.sign_: int = -1  # default to Mandelstam t convention (negative)

    def fit(self, x: np.ndarray) -> "LogStabilizedScaler":
        x = np.asarray(x, dtype=np.float64).ravel()
        x_abs = np.abs(x)
        median_abs = float(np.median(x_abs))
        # Guard: if the data happens to be concentrated at zero, don't let eps
        # collapse. We fall back to a small absolute floor.
        self.eps_ = max(self.t_epsilon_frac * median_abs, 1e-12)
        lx = np.log(x_abs + self.eps_)
        self.mean_ = float(np.mean(lx))
        std = float(np.std(lx))
        self.std_ = std if std > 0 else 1.0
        # Record dominant sign from training (majority vote, zeros excluded).
        signs = np.sign(x[x != 0])
        if signs.size > 0:
            self.sign_ = 1 if signs.sum() >= 0 else -1
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        self._check_fitted()
        x = np.asarray(x, dtype=np.float64)
        return (np.log(np.abs(x) + self.eps_) - self.mean_) / self.std_

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        """Invert the scaler. Sign is restored from the training-time majority
        sign (typically negative for Mandelstam t)."""
        self._check_fitted()
        x = np.asarray(x, dtype=np.float64)
        mag = np.exp(x * self.std_ + self.mean_) - self.eps_
        # Numerical floor: mag should be non-negative; clip tiny negatives.
        mag = np.clip(mag, 0.0, None)
        return self.sign_ * mag

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return self.fit(x).transform(x)

    def _check_fitted(self) -> None:
        if self.eps_ is None:
            raise RuntimeError(f"{type(self).__name__} used before fit()")


# =============================================================================
# Target scalers
# =============================================================================


class RobustTargetScaler:
    """``(y - median) / IQR``, robust to heavy-tailed outliers from
    diffractive dips."""

    def __init__(self) -> None:
        self.median_: float | None = None
        self.iqr_: float | None = None

    def fit(self, y: np.ndarray) -> "RobustTargetScaler":
        y = np.asarray(y, dtype=np.float64).ravel()
        self.median_ = float(np.median(y))
        q75, q25 = np.percentile(y, [75, 25])
        iqr = float(q75 - q25)
        self.iqr_ = iqr if iqr > 0 else 1.0
        return self

    def transform(self, y: np.ndarray) -> np.ndarray:
        self._check_fitted()
        y = np.asarray(y, dtype=np.float64)
        orig_shape = y.shape
        y = y.ravel()
        return ((y - self.median_) / self.iqr_).reshape(orig_shape)

    def inverse_transform(self, y: np.ndarray) -> np.ndarray:
        self._check_fitted()
        y = np.asarray(y, dtype=np.float64)
        return y * self.iqr_ + self.median_

    def fit_transform(self, y: np.ndarray) -> np.ndarray:
        return self.fit(y).transform(y)

    def std_to_original(self, std: np.ndarray | float) -> np.ndarray | float:
        """Scale a predictive std back to original units: std_orig = std * IQR."""
        self._check_fitted()
        return np.asarray(std) * self.iqr_

    def _check_fitted(self) -> None:
        if self.median_ is None:
            raise RuntimeError("RobustTargetScaler used before fit()")


class StandardTargetScaler:
    """``(y - mean) / std``. Alternative for cleaner distributions."""

    def __init__(self) -> None:
        self.mean_: float | None = None
        self.std_: float | None = None

    def fit(self, y: np.ndarray) -> "StandardTargetScaler":
        y = np.asarray(y, dtype=np.float64).ravel()
        self.mean_ = float(np.mean(y))
        std = float(np.std(y))
        self.std_ = std if std > 0 else 1.0
        return self

    def transform(self, y: np.ndarray) -> np.ndarray:
        self._check_fitted()
        y = np.asarray(y, dtype=np.float64)
        orig_shape = y.shape
        y = y.ravel()
        return ((y - self.mean_) / self.std_).reshape(orig_shape)

    def inverse_transform(self, y: np.ndarray) -> np.ndarray:
        self._check_fitted()
        y = np.asarray(y, dtype=np.float64)
        return y * self.std_ + self.mean_

    def fit_transform(self, y: np.ndarray) -> np.ndarray:
        return self.fit(y).transform(y)

    def std_to_original(self, std: np.ndarray | float) -> np.ndarray | float:
        self._check_fitted()
        return np.asarray(std) * self.std_

    def _check_fitted(self) -> None:
        if self.mean_ is None:
            raise RuntimeError("StandardTargetScaler used before fit()")


# =============================================================================
# Input scaler bundle
# =============================================================================


@dataclass
class InputScalerBundle:
    """Holds the three per-axis scalers and applies them as a group.

    ``fit``/``transform``/``inverse_transform`` accept ``X`` of shape (N, 3)
    with columns in the order (Q2, W2, t).
    """

    q2: object
    w2: object
    t: object

    def fit(self, X: np.ndarray) -> "InputScalerBundle":
        X = np.asarray(X, dtype=np.float64)
        self.q2.fit(X[:, 0])
        self.w2.fit(X[:, 1])
        self.t.fit(X[:, 2])
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        Xs = np.empty_like(X)
        Xs[:, 0] = self.q2.transform(X[:, 0])
        Xs[:, 1] = self.w2.transform(X[:, 1])
        Xs[:, 2] = self.t.transform(X[:, 2])
        return Xs

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        Xo = np.empty_like(X)
        Xo[:, 0] = self.q2.inverse_transform(X[:, 0])
        Xo[:, 1] = self.w2.inverse_transform(X[:, 1])
        Xo[:, 2] = self.t.inverse_transform(X[:, 2])
        return Xo

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


# =============================================================================
# Factories
# =============================================================================


def _build_axis_scaler(kind: str, is_prelogged: bool, t_eps_frac: float):
    """Map a config string to a scaler instance."""
    if kind == "zscore":
        return ZScoreScaler()
    if kind == "log_zscore":
        return ZScoreScaler() if is_prelogged else LogZScoreScaler()
    if kind == "log_stabilized":
        return LogStabilizedScaler(t_epsilon_frac=t_eps_frac)
    raise ValueError(f"Unknown axis transform '{kind}'")


def build_input_scaler(transforms_cfg, data_cfg) -> InputScalerBundle:
    """Assemble an ``InputScalerBundle`` from a ``TransformsConfig``.

    Q2 respects ``data_cfg.q2_is_prelogged``: if True, ``log_zscore`` collapses
    to plain ``zscore``. W2 is always a raw positive value in our schema.
    """
    q2 = _build_axis_scaler(
        transforms_cfg.q2_transform,
        is_prelogged=data_cfg.q2_is_prelogged,
        t_eps_frac=transforms_cfg.t_epsilon_frac,
    )
    w2 = _build_axis_scaler(
        transforms_cfg.w2_transform,
        is_prelogged=False,
        t_eps_frac=transforms_cfg.t_epsilon_frac,
    )
    t = _build_axis_scaler(
        transforms_cfg.t_transform,
        is_prelogged=False,
        t_eps_frac=transforms_cfg.t_epsilon_frac,
    )
    return InputScalerBundle(q2=q2, w2=w2, t=t)


def build_target_scaler(transforms_cfg):
    """Return the target scaler specified in config."""
    if transforms_cfg.target_scaler == "robust":
        return RobustTargetScaler()
    if transforms_cfg.target_scaler == "standard":
        return StandardTargetScaler()
    raise ValueError(f"Unknown target_scaler '{transforms_cfg.target_scaler}'")
