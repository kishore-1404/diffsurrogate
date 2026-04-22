"""Matplotlib plots for benchmark results.

Three plots per model:

  residuals_{model}.png     — scatter of prediction error vs |t|
  uq_bands_{model}.png      — predictions with ±2σ bands vs test index
                              (only generated when the model supports UQ)
  t_spectrum_{model}.png    — truth vs prediction sweep over t at a fixed
                              (Q², W²) slice chosen from the test set

All plotting uses a headless ``Agg`` backend so it works in CI / servers.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

logger = logging.getLogger(__name__)


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    t_values: np.ndarray,
    model_name: str,
    out_path: Path,
) -> None:
    """Scatter of residuals vs |t| on a log scale."""
    resid = np.asarray(y_true).ravel() - np.asarray(y_pred).ravel()
    t_abs = np.abs(np.asarray(t_values).ravel())

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(t_abs, resid, s=8, alpha=0.6)
    ax.axhline(0.0, color="k", linewidth=0.8, linestyle="--")
    ax.set_xscale("log")
    ax.set_xlabel(r"$|t|$")
    ax.set_ylabel("residual (scaled)")
    ax.set_title(f"Residuals: {model_name}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_uq_bands(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_std: np.ndarray,
    model_name: str,
    out_path: Path,
    max_points: int = 300,
) -> None:
    """Line plot of predictions ± 2σ bands with truth overlaid, sorted by truth."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    y_std = np.asarray(y_std).ravel()

    order = np.argsort(y_true)
    if order.size > max_points:
        # subsample evenly for readability
        idx = np.linspace(0, order.size - 1, max_points).astype(int)
        order = order[idx]

    xs = np.arange(order.size)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(xs, y_true[order], "k.", markersize=3, label="truth")
    ax.plot(xs, y_pred[order], "C0-", linewidth=1.0, label="pred")
    ax.fill_between(
        xs,
        y_pred[order] - 2.0 * y_std[order],
        y_pred[order] + 2.0 * y_std[order],
        color="C0",
        alpha=0.2,
        label=r"$\pm 2\sigma$",
    )
    ax.set_xlabel("test index (sorted by truth)")
    ax.set_ylabel("target (scaled)")
    ax.set_title(f"UQ bands: {model_name}")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_t_spectrum(
    X_test: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    out_path: Path,
    t_col_index: int = 2,
    tol: float = 1e-6,
) -> None:
    """Overlay truth vs prediction across t at a fixed (Q2, W2) slice.

    We pick the (Q2, W2) combination in the test set that has the most
    distinct t-points so that the overlay is informative even on sparse data.
    """
    X = np.asarray(X_test)
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    # Identify (Q2, W2) slices.
    q = np.round(X[:, 0], decimals=6)
    w = np.round(X[:, 1], decimals=6)
    keys = np.stack([q, w], axis=1)
    uniq, inv, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    if uniq.shape[0] == 0:
        return
    best = int(np.argmax(counts))
    mask = inv == best
    if mask.sum() < 2:
        # Fallback: just plot the full test set sorted by t.
        mask = np.ones_like(inv, dtype=bool)

    t_slice = X[mask, t_col_index]
    order = np.argsort(t_slice)
    t_sorted = t_slice[order]
    yt = y_true[mask][order]
    yp = y_pred[mask][order]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(np.abs(t_sorted), yt, "ko-", markersize=4, linewidth=1.0, label="truth")
    ax.plot(np.abs(t_sorted), yp, "C3x--", markersize=5, linewidth=1.0, label="pred")
    ax.set_xscale("log")
    ax.set_xlabel(r"$|t|$")
    ax.set_ylabel("target (scaled)")
    slice_label = (
        f"Q2={uniq[best, 0]:.3g}, W2={uniq[best, 1]:.3g}"
        if mask.sum() > 1 and mask.sum() < X.shape[0]
        else "all test points"
    )
    ax.set_title(f"t-spectrum: {model_name}  ({slice_label})")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
