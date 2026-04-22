"""Evaluation metrics.

Pure functions that accept NumPy arrays and return scalars. All metrics are
NaN-safe: if inputs contain NaNs we drop those rows and emit a single
``logging.warning``, rather than propagating NaN or raising.

Exposed metrics:

  rmse(y_true, y_pred)                       → float
  mae(y_true, y_pred)                        → float
  nll(y_true, y_pred, y_std)                 → float (Gaussian NLL; NaN if no std)
  coverage_95(y_true, y_pred, y_std)         → float (fraction inside ±2σ)
  constraint_violation_rate(y_physical, ...) → float (fraction violating 0≤N≤1)

``compute_metrics`` is a dispatcher that runs whichever metrics were listed
in the ``[evaluation] metrics`` config section.
"""

from __future__ import annotations

import logging
import math
import warnings
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Tolerance for treating predicted amplitudes as "clearly unphysical" beyond
# the unitarity bound. 0 and 1 are exact bounds on N; we leave a tiny float
# slack for numerical reasons.
_UNITARITY_EPS = 1e-9


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _finite_mask(*arrays: np.ndarray) -> np.ndarray:
    """Elementwise mask of rows where every given array is finite."""
    mask = np.ones(arrays[0].shape[0], dtype=bool)
    for a in arrays:
        mask &= np.isfinite(a)
    return mask


def _warn_nan(name: str, n_dropped: int, n_total: int) -> None:
    if n_dropped == 0:
        return
    warnings.warn(
        f"{name}: dropped {n_dropped}/{n_total} rows containing NaN/inf",
        RuntimeWarning,
        stacklevel=3,
    )
    logger.warning("%s: dropped %d/%d non-finite rows", name, n_dropped, n_total)


# -----------------------------------------------------------------------------
# Individual metrics
# -----------------------------------------------------------------------------

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error."""
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    mask = _finite_mask(y_true, y_pred)
    _warn_nan("rmse", y_true.size - mask.sum(), y_true.size)
    if mask.sum() == 0:
        return float("nan")
    diff = y_true[mask] - y_pred[mask]
    return float(np.sqrt(np.mean(diff * diff)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error."""
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    mask = _finite_mask(y_true, y_pred)
    _warn_nan("mae", y_true.size - mask.sum(), y_true.size)
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask])))


def nll(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_std: np.ndarray | None,
) -> float:
    """Gaussian negative log-likelihood per sample.

    Returns NaN if ``y_std`` is None (model doesn't support UQ). Non-positive
    stds are floored to a tiny positive number to keep the log defined.
    """
    if y_std is None:
        return float("nan")
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    y_std = np.asarray(y_std, dtype=np.float64).ravel()
    mask = _finite_mask(y_true, y_pred, y_std)
    _warn_nan("nll", y_true.size - mask.sum(), y_true.size)
    if mask.sum() == 0:
        return float("nan")
    yt = y_true[mask]
    yp = y_pred[mask]
    ys = np.maximum(y_std[mask], 1e-12)
    # NLL = 0.5 * ((yt - yp)^2 / ys^2 + log(2π ys^2))
    term = (yt - yp) ** 2 / (ys ** 2) + np.log(2.0 * math.pi * ys ** 2)
    return float(0.5 * np.mean(term))


def coverage_95(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_std: np.ndarray | None,
) -> float:
    """Fraction of points where |y_true - y_pred| ≤ 2·y_std (empirical 95% CI).

    NaN if ``y_std`` is None.
    """
    if y_std is None:
        return float("nan")
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    y_std = np.asarray(y_std, dtype=np.float64).ravel()
    mask = _finite_mask(y_true, y_pred, y_std)
    _warn_nan("coverage_95", y_true.size - mask.sum(), y_true.size)
    if mask.sum() == 0:
        return float("nan")
    inside = np.abs(y_true[mask] - y_pred[mask]) <= 2.0 * np.maximum(y_std[mask], 0.0)
    return float(np.mean(inside))


def constraint_violation_rate(
    y_pred_physical: np.ndarray | None,
    n_pred: np.ndarray | None = None,
) -> float:
    """Fraction of predictions that violate the unitarity bound 0 ≤ N ≤ 1.

    Spec note: our surrogates predict ln(A), not N directly. Only the PINN
    exposes an auxiliary N head. The resolution we adopt:

      - If an explicit ``n_pred`` array is provided (e.g. from a PINN's
        ``N_head``), we check ``0 - eps ≤ N ≤ 1 + eps`` directly.
      - Otherwise we treat ``y_pred_physical`` as A (the amplitude returned
        from the inverse target-scaler and then exp'd back from ln(A)), and
        report the fraction where A > 1 or A < 0 as a coarse proxy for
        unitarity violation. Given that A is a probability-like quantity
        bounded by the black-disk limit, this is the closest readily
        available check without special-casing each model.
      - If ``y_pred_physical`` is also None (model skipped entirely), we
        return NaN.
    """
    if n_pred is not None:
        arr = np.asarray(n_pred, dtype=np.float64).ravel()
        mask = np.isfinite(arr)
        _warn_nan("constraint_violation_rate", arr.size - mask.sum(), arr.size)
        if mask.sum() == 0:
            return float("nan")
        arr = arr[mask]
        violated = (arr < -_UNITARITY_EPS) | (arr > 1.0 + _UNITARITY_EPS)
        return float(np.mean(violated))

    if y_pred_physical is None:
        return float("nan")
    arr = np.asarray(y_pred_physical, dtype=np.float64).ravel()
    mask = np.isfinite(arr)
    _warn_nan("constraint_violation_rate", arr.size - mask.sum(), arr.size)
    if mask.sum() == 0:
        return float("nan")
    arr = arr[mask]
    violated = (arr < -_UNITARITY_EPS) | (arr > 1.0 + _UNITARITY_EPS)
    return float(np.mean(violated))


# -----------------------------------------------------------------------------
# Dispatcher
# -----------------------------------------------------------------------------

_DISPATCH = {
    "rmse": "rmse",
    "mae": "mae",
    "nll": "nll",
    "coverage_95": "coverage_95",
    "constraint_violation_rate": "constraint_violation_rate",
}


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_std: np.ndarray | None = None,
    y_pred_physical: np.ndarray | None = None,
    n_pred: np.ndarray | None = None,
    enabled: list[str] | None = None,
) -> dict[str, Any]:
    """Run every metric in ``enabled`` and return a dict name→scalar.

    Parameters
    ----------
    y_true, y_pred : arrays in scaled space (the space where the model was
        trained). These are what you want to measure RMSE, NLL, and coverage
        on.
    y_std : predictive std in scaled space, or None.
    y_pred_physical : predictions converted back to physical amplitude units.
        Used only by ``constraint_violation_rate``.
    n_pred : optional direct N predictions (e.g. from a PINN N-head) for the
        constraint-violation metric. Preferred over ``y_pred_physical`` when
        available.
    enabled : list of metric names. Defaults to all.

    Returns
    -------
    dict : mapping of metric name → float (possibly NaN).
    """
    if enabled is None:
        enabled = list(_DISPATCH)
    out: dict[str, Any] = {}
    for name in enabled:
        if name not in _DISPATCH:
            logger.warning("Unknown metric '%s' — skipping", name)
            out[name] = float("nan")
            continue
        if name == "rmse":
            out[name] = rmse(y_true, y_pred)
        elif name == "mae":
            out[name] = mae(y_true, y_pred)
        elif name == "nll":
            out[name] = nll(y_true, y_pred, y_std)
        elif name == "coverage_95":
            out[name] = coverage_95(y_true, y_pred, y_std)
        elif name == "constraint_violation_rate":
            out[name] = constraint_violation_rate(y_pred_physical, n_pred)
    return out
