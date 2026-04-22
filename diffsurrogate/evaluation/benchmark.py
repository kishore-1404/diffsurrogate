"""Benchmark orchestrator.

Implements Mode 1 from the spec: compare all enabled surrogates head-to-head
against a held-out test set, writing a CSV/JSON report and plots.

Pipeline per model:

    1. Instantiate with its config sub-table
    2. Fit on the training split (in *scaled* space)
    3. Predict on the test split
    4. Compute configured metrics (scaled space for RMSE/NLL/coverage,
       physical space for constraint violation)
    5. Save model + scalers under model_dir/{name}/benchmark/
    6. Plot residuals / UQ bands / t-spectrum
"""

from __future__ import annotations

import logging
import math
import time
import warnings
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from diffsurrogate.config import Config
from diffsurrogate.data import build_input_scaler, build_target_scaler, load_dataset
from diffsurrogate.data.splitter import stratified_split_on_t
from diffsurrogate.evaluation.metrics import compute_metrics
from diffsurrogate.evaluation.report import print_leaderboard, write_report
from diffsurrogate.evaluation.visualize import (
    plot_residuals,
    plot_t_spectrum,
    plot_uq_bands,
)
from diffsurrogate.models import build_model
from diffsurrogate.persistence.saver import save_model

logger = logging.getLogger(__name__)


# Models that produce a meaningful std (coverage_95 / NLL are only reported
# as non-NaN for these).
_UQ_MODELS = {"gaussian_process", "deep_gp", "pce"}


def run_benchmark(cfg: Config) -> list[dict[str, Any]]:
    """Run the full Mode-1 benchmark pipeline.

    Returns the per-model results list (also written to disk as CSV/JSON).
    """
    # --- Load data ---
    _, X, y = load_dataset(cfg.data.input_path, cfg.data.input_columns, cfg.data.target_column)

    # --- Train/test split (stratified on |t|) ---
    X_train, X_test, y_train, y_test = stratified_split_on_t(
        X, y,
        train_fraction=cfg.data.train_fraction,
        random_seed=cfg.data.random_seed,
    )

    # --- Fit transforms on training data only ---
    input_scaler = build_input_scaler(cfg.transforms, cfg.data)
    target_scaler = build_target_scaler(cfg.transforms)
    X_train_s = input_scaler.fit_transform(X_train)
    X_test_s = input_scaler.transform(X_test)
    y_train_s = target_scaler.fit_transform(y_train).ravel()
    y_test_s = target_scaler.transform(y_test).ravel()

    # --- Prepare output dirs ---
    output_dir = Path(cfg.evaluation.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    model_dir = Path(cfg.persistence.model_dir)

    # --- Run each enabled model ---
    results: list[dict[str, Any]] = []
    for name in cfg.models.enabled:
        sub_cfg = cfg.models.get(name)
        if sub_cfg is None:
            logger.warning("Model '%s' is enabled but has no [models.%s] sub-table — skipping", name, name)
            continue

        logger.info("=" * 60)
        logger.info("BENCHMARK: %s", name)
        logger.info("=" * 60)
        try:
            row = _fit_eval_one(
                name=name,
                sub_cfg=sub_cfg,
                cfg=cfg,
                X_train_s=X_train_s, y_train_s=y_train_s,
                X_test=X_test, X_test_s=X_test_s,
                y_test=y_test, y_test_s=y_test_s,
                input_scaler=input_scaler, target_scaler=target_scaler,
                model_dir=model_dir, plots_dir=plots_dir,
            )
            results.append(row)
        except ImportError as e:
            logger.warning("Model '%s' skipped due to missing dependency: %s", name, e)
            results.append({
                "model": name, "error": f"missing dependency: {e}",
                **{m: float("nan") for m in cfg.evaluation.metrics},
                "fit_time_sec": float("nan"),
                "predict_time_sec": float("nan"),
                "n_train": X_train.shape[0], "n_test": X_test.shape[0],
                "supports_uq": False,
            })
        except Exception as e:  # noqa: BLE001
            logger.exception("Model '%s' failed during benchmark: %s", name, e)
            results.append({
                "model": name, "error": str(e),
                **{m: float("nan") for m in cfg.evaluation.metrics},
                "fit_time_sec": float("nan"),
                "predict_time_sec": float("nan"),
                "n_train": X_train.shape[0], "n_test": X_test.shape[0],
                "supports_uq": False,
            })

    # --- Report ---
    write_report(results, output_dir, cfg.evaluation.metrics)
    print_leaderboard(results, cfg.evaluation.metrics)
    return results


def _fit_eval_one(
    *,
    name: str,
    sub_cfg: Any,
    cfg: Config,
    X_train_s: np.ndarray, y_train_s: np.ndarray,
    X_test: np.ndarray, X_test_s: np.ndarray,
    y_test: np.ndarray, y_test_s: np.ndarray,
    input_scaler, target_scaler,
    model_dir: Path, plots_dir: Path,
) -> dict[str, Any]:
    """Fit, predict, compute metrics, save artifacts, and plot for one model."""
    model = build_model(name, sub_cfg)

    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X_train_s, y_train_s)
    fit_time = time.time() - t0

    t0 = time.time()
    mu_s, std_s = model.predict(X_test_s)
    predict_time = time.time() - t0

    # Physical-space mean (for constraint metric). ln(A) → A.
    mu_phys_lnA = target_scaler.inverse_transform(mu_s)
    with np.errstate(over="ignore"):
        mu_phys = np.exp(mu_phys_lnA)

    # Metrics.
    metrics_out = compute_metrics(
        y_true=y_test_s,
        y_pred=mu_s,
        y_std=std_s,
        y_pred_physical=mu_phys,
        enabled=cfg.evaluation.metrics,
    )

    # Plots.
    if cfg.evaluation.save_plots:
        try:
            plot_residuals(
                y_test_s, mu_s, X_test[:, 2], name,
                plots_dir / f"residuals_{name}.png",
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("residual plot for %s failed: %s", name, e)
        if std_s is not None and name in _UQ_MODELS:
            try:
                plot_uq_bands(
                    y_test_s, mu_s, std_s, name,
                    plots_dir / f"uq_bands_{name}.png",
                )
            except Exception as e:  # noqa: BLE001
                logger.warning("UQ plot for %s failed: %s", name, e)
        try:
            plot_t_spectrum(
                X_test, y_test_s, mu_s, name,
                plots_dir / f"t_spectrum_{name}.png",
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("t-spectrum plot for %s failed: %s", name, e)

    # Persist model + scalers under benchmark/.
    try:
        save_dir = model_dir / name / "benchmark"
        save_model(
            model,
            {"input_scaler": input_scaler, "target_scaler": target_scaler},
            save_dir,
            n_train=int(X_train_s.shape[0]),
            config_snapshot={"model": asdict(sub_cfg) if hasattr(sub_cfg, "__dataclass_fields__") else {},
                             "transforms": asdict(cfg.transforms),
                             "data": asdict(cfg.data)},
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("save_model for %s failed: %s", name, e)

    return {
        "model": name,
        **metrics_out,
        "fit_time_sec": float(fit_time),
        "predict_time_sec": float(predict_time),
        "n_train": int(X_train_s.shape[0]),
        "n_test": int(X_test_s.shape[0]),
        "supports_uq": bool(model.supports_uq()),
    }
