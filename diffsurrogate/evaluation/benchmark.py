"""Benchmark orchestrator."""

from __future__ import annotations

import logging
import time
import warnings
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from diffsurrogate.config import Config
from diffsurrogate.data import (
    build_input_scaler,
    build_target_scaler,
    load_dataset,
    stratified_split_indices_on_t,
)
from diffsurrogate.evaluation.artifacts import (
    create_run_layout,
    write_model_failure,
    write_prediction_artifact,
    write_research_prompt_pack,
    write_run_manifest,
    write_split_artifacts,
)
from diffsurrogate.evaluation.metrics import compute_metrics
from diffsurrogate.evaluation.report import print_leaderboard, write_report
from diffsurrogate.evaluation.visualize import plot_residuals, plot_t_spectrum, plot_uq_bands
from diffsurrogate.models import build_model
from diffsurrogate.persistence.saver import artifact_size_bytes, save_model

logger = logging.getLogger(__name__)

_UQ_MODELS = {'gaussian_process', 'deep_gp', 'pce'}


def run_benchmark(cfg: Config) -> list[dict[str, Any]]:
    """Run the full Mode-1 benchmark pipeline."""
    df, X, y = load_dataset(cfg.data.input_path, cfg.data.input_columns, cfg.data.target_column)
    train_idx, test_idx = stratified_split_indices_on_t(
        X,
        train_fraction=cfg.data.train_fraction,
        random_seed=cfg.data.random_seed,
    )

    X_train = X[train_idx]
    X_test = X[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]
    df_test = df.iloc[test_idx].copy().reset_index(drop=True)

    input_scaler = build_input_scaler(cfg.transforms, cfg.data)
    target_scaler = build_target_scaler(cfg.transforms)
    X_train_s = input_scaler.fit_transform(X_train)
    X_test_s = input_scaler.transform(X_test)
    y_train_s = target_scaler.fit_transform(y_train).ravel()
    y_test_s = target_scaler.transform(y_test).ravel()

    layout = create_run_layout(cfg)
    output_dir = Path(layout['output_dir'])
    run_dir = Path(layout['run_dir'])
    plots_dir = Path(layout['plots_dir'])
    reports_dir = Path(layout['reports_dir'])
    predictions_dir = Path(layout['predictions_dir'])
    metadata_dir = Path(layout['metadata_dir'])
    splits_dir = Path(layout['splits_dir'])
    run_id = str(layout['run_id'])
    model_dir = Path(cfg.persistence.model_dir)

    if cfg.evaluation.save_run_manifest:
        write_run_manifest(
            cfg=cfg,
            df=df,
            train_idx=train_idx,
            test_idx=test_idx,
            run_id=run_id,
            run_dir=run_dir,
            model_names=list(cfg.models.enabled),
        )
    if cfg.evaluation.save_split_data:
        write_split_artifacts(df=df, train_idx=train_idx, test_idx=test_idx, splits_dir=splits_dir)

    results: list[dict[str, Any]] = []
    for name in cfg.models.enabled:
        sub_cfg = cfg.models.get(name)
        if sub_cfg is None:
            logger.warning("Model '%s' is enabled but has no [models.%s] sub-table; skipping", name, name)
            row = _failure_row(
                name=name,
                metrics=cfg.evaluation.metrics,
                n_train=X_train.shape[0],
                n_test=X_test.shape[0],
                status='invalid_config',
                error=f'missing [models.{name}] sub-table',
            )
            write_model_failure(metadata_dir=metadata_dir, model_name=name, status=row['status'], error=row['error'])
            results.append(row)
            continue

        logger.info('=' * 60)
        logger.info('BENCHMARK: %s', name)
        logger.info('=' * 60)
        try:
            row = _fit_eval_one(
                name=name,
                sub_cfg=sub_cfg,
                cfg=cfg,
                X_train_s=X_train_s,
                y_train_s=y_train_s,
                X_test=X_test,
                X_test_s=X_test_s,
                y_test=y_test,
                y_test_s=y_test_s,
                df_test=df_test,
                test_idx=test_idx,
                input_scaler=input_scaler,
                target_scaler=target_scaler,
                model_dir=model_dir,
                plots_dir=plots_dir,
                predictions_dir=predictions_dir,
                run_id=run_id,
            )
            results.append(row)
        except ImportError as exc:
            logger.warning("Model '%s' skipped due to missing dependency: %s", name, exc)
            row = _failure_row(
                name=name,
                metrics=cfg.evaluation.metrics,
                n_train=X_train.shape[0],
                n_test=X_test.shape[0],
                status='missing_dependency',
                error=f'missing dependency: {exc}',
            )
            write_model_failure(metadata_dir=metadata_dir, model_name=name, status=row['status'], error=row['error'])
            results.append(row)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Model '%s' failed during benchmark: %s", name, exc)
            row = _failure_row(
                name=name,
                metrics=cfg.evaluation.metrics,
                n_train=X_train.shape[0],
                n_test=X_test.shape[0],
                status='failed',
                error=str(exc),
            )
            write_model_failure(metadata_dir=metadata_dir, model_name=name, status=row['status'], error=row['error'])
            results.append(row)

    write_report(results, output_dir, cfg.evaluation.metrics, write_markdown=cfg.evaluation.write_markdown_report)
    write_report(results, reports_dir, cfg.evaluation.metrics, write_markdown=cfg.evaluation.write_markdown_report)
    write_research_prompt_pack(
        reports_dir=reports_dir,
        results=results,
        metrics=cfg.evaluation.metrics,
        model_names=list(cfg.models.enabled),
        run_id=run_id,
    )
    print_leaderboard(results, cfg.evaluation.metrics)
    logger.info('Benchmark run bundle: %s', run_dir)
    return results


def _fit_eval_one(
    *,
    name: str,
    sub_cfg: Any,
    cfg: Config,
    X_train_s: np.ndarray,
    y_train_s: np.ndarray,
    X_test: np.ndarray,
    X_test_s: np.ndarray,
    y_test: np.ndarray,
    y_test_s: np.ndarray,
    df_test: pd.DataFrame,
    test_idx: np.ndarray,
    input_scaler,
    target_scaler,
    model_dir: Path,
    plots_dir: Path,
    predictions_dir: Path,
    run_id: str,
) -> dict[str, Any]:
    """Fit, evaluate, save artifacts, and emit per-model prediction dumps."""
    model = build_model(name, sub_cfg)

    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        model.fit(X_train_s, y_train_s)
    fit_time = time.time() - t0

    t0 = time.time()
    mu_s, std_s = model.predict(X_test_s)
    predict_time = time.time() - t0

    mu_s = np.asarray(mu_s, dtype=np.float64).ravel()
    std_s_arr = None if std_s is None else np.asarray(std_s, dtype=np.float64).ravel()
    y_test_lnA = np.asarray(y_test, dtype=np.float64).ravel()
    mu_lnA = np.asarray(target_scaler.inverse_transform(mu_s), dtype=np.float64).ravel()
    with np.errstate(over='ignore'):
        mu_phys = np.exp(mu_lnA)
        y_true_phys = np.exp(y_test_lnA)

    metrics_out = compute_metrics(
        y_true=y_test_s,
        y_pred=mu_s,
        y_std=std_s_arr,
        y_pred_physical=mu_phys,
        enabled=cfg.evaluation.metrics,
    )
    diagnostics = _prediction_diagnostics(y_true=y_test_s, y_pred=mu_s)

    plot_paths: list[str] = []
    if cfg.evaluation.save_plots:
        residual_path = plots_dir / f'residuals_{name}.png'
        try:
            plot_residuals(y_test_s, mu_s, X_test[:, 2], name, residual_path)
            plot_paths.append(str(residual_path))
        except Exception as exc:  # noqa: BLE001
            logger.warning('residual plot for %s failed: %s', name, exc)
        if std_s_arr is not None and name in _UQ_MODELS:
            uq_path = plots_dir / f'uq_bands_{name}.png'
            try:
                plot_uq_bands(y_test_s, mu_s, std_s_arr, name, uq_path)
                plot_paths.append(str(uq_path))
            except Exception as exc:  # noqa: BLE001
                logger.warning('UQ plot for %s failed: %s', name, exc)
        t_path = plots_dir / f't_spectrum_{name}.png'
        try:
            plot_t_spectrum(X_test, y_test_s, mu_s, name, t_path)
            plot_paths.append(str(t_path))
        except Exception as exc:  # noqa: BLE001
            logger.warning('t-spectrum plot for %s failed: %s', name, exc)

    artifact_dir = model_dir / name / 'benchmark'
    artifact_size = 0
    try:
        save_model(
            model,
            {'input_scaler': input_scaler, 'target_scaler': target_scaler},
            artifact_dir,
            n_train=int(X_train_s.shape[0]),
            config_snapshot={
                'run_id': run_id,
                'model': asdict(sub_cfg) if hasattr(sub_cfg, '__dataclass_fields__') else {},
                'transforms': asdict(cfg.transforms),
                'data': asdict(cfg.data),
                'evaluation': asdict(cfg.evaluation),
            },
        )
        artifact_size = artifact_size_bytes(artifact_dir)
    except Exception as exc:  # noqa: BLE001
        logger.warning('save_model for %s failed: %s', name, exc)

    predictions_path = None
    if cfg.evaluation.save_predictions:
        predictions_df = _build_prediction_frame(
            df_test=df_test,
            dataset_indices=test_idx,
            y_true_scaled=y_test_s,
            y_pred_scaled=mu_s,
            y_true_lnA=y_test_lnA,
            y_pred_lnA=mu_lnA,
            y_true_phys=y_true_phys,
            y_pred_phys=mu_phys,
            y_std_scaled=std_s_arr,
            target_scaler=target_scaler,
        )
        predictions_path = write_prediction_artifact(predictions_df, predictions_dir=predictions_dir, model_name=name)

    return {
        'model': name,
        'status': 'ok',
        'error': None,
        **metrics_out,
        **diagnostics,
        'fit_time_sec': float(fit_time),
        'predict_time_sec': float(predict_time),
        'n_train': int(X_train_s.shape[0]),
        'n_test': int(X_test_s.shape[0]),
        'supports_uq': bool(model.supports_uq()),
        'artifact_size_bytes': int(artifact_size),
        'artifact_dir': str(artifact_dir),
        'predictions_path': str(predictions_path) if predictions_path is not None else None,
        'plot_paths': plot_paths,
    }


def _build_prediction_frame(
    *,
    df_test: pd.DataFrame,
    dataset_indices: np.ndarray,
    y_true_scaled: np.ndarray,
    y_pred_scaled: np.ndarray,
    y_true_lnA: np.ndarray,
    y_pred_lnA: np.ndarray,
    y_true_phys: np.ndarray,
    y_pred_phys: np.ndarray,
    y_std_scaled: np.ndarray | None,
    target_scaler,
) -> pd.DataFrame:
    out = df_test.copy()
    out.insert(0, 'dataset_index', np.asarray(dataset_indices, dtype=int))
    out['true_scaled_target'] = np.asarray(y_true_scaled, dtype=np.float64).ravel()
    out['predicted_scaled_target'] = np.asarray(y_pred_scaled, dtype=np.float64).ravel()
    out['scaled_residual'] = out['true_scaled_target'] - out['predicted_scaled_target']
    out['abs_scaled_residual'] = np.abs(out['scaled_residual'])
    out['true_ln_amplitude'] = np.asarray(y_true_lnA, dtype=np.float64).ravel()
    out['predicted_ln_amplitude'] = np.asarray(y_pred_lnA, dtype=np.float64).ravel()
    out['ln_amplitude_residual'] = out['true_ln_amplitude'] - out['predicted_ln_amplitude']
    out['true_amplitude'] = np.asarray(y_true_phys, dtype=np.float64).ravel()
    out['predicted_amplitude'] = np.asarray(y_pred_phys, dtype=np.float64).ravel()
    out['amplitude_residual'] = out['true_amplitude'] - out['predicted_amplitude']

    if y_std_scaled is not None:
        out['std_scaled_target'] = np.asarray(y_std_scaled, dtype=np.float64).ravel()
        if hasattr(target_scaler, 'std_to_original'):
            std_ln = np.asarray(target_scaler.std_to_original(y_std_scaled), dtype=np.float64).ravel()
        else:
            std_ln = np.asarray(y_std_scaled, dtype=np.float64).ravel()
        out['std_ln_amplitude'] = std_ln
        lo_ln = out['predicted_ln_amplitude'] - 2.0 * out['std_ln_amplitude']
        hi_ln = out['predicted_ln_amplitude'] + 2.0 * out['std_ln_amplitude']
        out['lower_2sigma_ln_amplitude'] = lo_ln
        out['upper_2sigma_ln_amplitude'] = hi_ln
        with np.errstate(over='ignore'):
            out['lower_2sigma_amplitude'] = np.exp(lo_ln)
            out['upper_2sigma_amplitude'] = np.exp(hi_ln)
        out['inside_2sigma_interval'] = (
            (out['true_ln_amplitude'] >= out['lower_2sigma_ln_amplitude'])
            & (out['true_ln_amplitude'] <= out['upper_2sigma_ln_amplitude'])
        )
    return out


def _prediction_diagnostics(*, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    residual = np.asarray(y_true, dtype=np.float64).ravel() - np.asarray(y_pred, dtype=np.float64).ravel()
    abs_residual = np.abs(residual)
    return {
        'residual_mean': float(np.mean(residual)),
        'residual_std': float(np.std(residual)),
        'abs_residual_p50': float(np.quantile(abs_residual, 0.50)),
        'abs_residual_p90': float(np.quantile(abs_residual, 0.90)),
        'max_abs_residual': float(np.max(abs_residual)),
    }


def _failure_row(*, name: str, metrics: list[str], n_train: int, n_test: int, status: str, error: str) -> dict[str, Any]:
    row = {
        'model': name,
        'status': status,
        'error': error,
        'fit_time_sec': float('nan'),
        'predict_time_sec': float('nan'),
        'n_train': int(n_train),
        'n_test': int(n_test),
        'supports_uq': False,
        'artifact_size_bytes': 0,
        'artifact_dir': None,
        'predictions_path': None,
        'residual_mean': float('nan'),
        'residual_std': float('nan'),
        'abs_residual_p50': float('nan'),
        'abs_residual_p90': float('nan'),
        'max_abs_residual': float('nan'),
    }
    for metric in metrics:
        row[metric] = float('nan')
    return row
