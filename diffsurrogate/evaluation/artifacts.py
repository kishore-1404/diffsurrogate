"""Benchmark artifact helpers."""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from diffsurrogate.config import Config
from diffsurrogate.evaluation.research import build_research_prompts, method_rows


def _json_safe(value: Any) -> Any:
    if is_dataclass(value):
        return {k: _json_safe(v) for k, v in asdict(value).items()}
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def create_run_layout(cfg: Config) -> dict[str, Path | str]:
    output_dir = Path(cfg.evaluation.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base_run_id = time.strftime('%Y%m%d_%H%M%S')
    runs_root = output_dir / cfg.evaluation.benchmark_runs_dirname
    run_id = base_run_id
    run_dir = runs_root / run_id
    suffix = 1
    while run_dir.exists():
        run_id = f'{base_run_id}_{suffix:02d}'
        run_dir = runs_root / run_id
        suffix += 1
    reports_dir = run_dir / 'reports'
    predictions_dir = run_dir / 'predictions'
    metadata_dir = run_dir / 'metadata'
    splits_dir = run_dir / 'splits'
    plots_dir = run_dir / 'plots'
    for directory in (run_dir, reports_dir, predictions_dir, metadata_dir, splits_dir, plots_dir):
        directory.mkdir(parents=True, exist_ok=True)
    return {
        'run_id': run_id,
        'output_dir': output_dir,
        'run_dir': run_dir,
        'reports_dir': reports_dir,
        'predictions_dir': predictions_dir,
        'metadata_dir': metadata_dir,
        'splits_dir': splits_dir,
        'plots_dir': plots_dir,
    }


def write_run_manifest(
    *,
    cfg: Config,
    df: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    run_id: str,
    run_dir: Path,
    model_names: list[str],
) -> Path:
    manifest = {
        'run_id': run_id,
        'created_at': time.strftime('%Y-%m-%dT%H:%M:%S%z') or time.strftime('%Y-%m-%dT%H:%M:%S'),
        'dataset': {
            'input_path': cfg.data.input_path,
            'n_rows': int(df.shape[0]),
            'input_columns': list(cfg.data.input_columns),
            'target_column': cfg.data.target_column,
        },
        'split': {
            'train_fraction': cfg.data.train_fraction,
            'random_seed': cfg.data.random_seed,
            'n_train': int(train_idx.size),
            'n_test': int(test_idx.size),
            'train_indices_path': 'splits/train_indices.csv',
            'test_indices_path': 'splits/test_indices.csv',
            't_abs_summary': {
                'train_min': float(np.min(np.abs(df.iloc[train_idx][cfg.data.input_columns[2]].to_numpy(dtype=float)))) if train_idx.size else None,
                'train_max': float(np.max(np.abs(df.iloc[train_idx][cfg.data.input_columns[2]].to_numpy(dtype=float)))) if train_idx.size else None,
                'test_min': float(np.min(np.abs(df.iloc[test_idx][cfg.data.input_columns[2]].to_numpy(dtype=float)))) if test_idx.size else None,
                'test_max': float(np.max(np.abs(df.iloc[test_idx][cfg.data.input_columns[2]].to_numpy(dtype=float)))) if test_idx.size else None,
            },
        },
        'evaluation': {
            'metrics': list(cfg.evaluation.metrics),
            'save_plots': cfg.evaluation.save_plots,
            'save_predictions': cfg.evaluation.save_predictions,
            'save_split_data': cfg.evaluation.save_split_data,
            'write_markdown_report': cfg.evaluation.write_markdown_report,
        },
        'models_enabled': list(model_names),
        'method_cards': method_rows(model_names),
        'config_snapshot': _json_safe(cfg.raw),
    }
    path = run_dir / 'metadata' / 'run_manifest.json'
    path.write_text(json.dumps(_json_safe(manifest), indent=2))
    return path


def write_split_artifacts(
    *,
    df: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    splits_dir: Path,
) -> dict[str, Path]:
    train_idx_df = pd.DataFrame({'dataset_index': np.asarray(train_idx, dtype=int)})
    test_idx_df = pd.DataFrame({'dataset_index': np.asarray(test_idx, dtype=int)})
    train_idx_path = splits_dir / 'train_indices.csv'
    test_idx_path = splits_dir / 'test_indices.csv'
    train_idx_df.to_csv(train_idx_path, index=False)
    test_idx_df.to_csv(test_idx_path, index=False)

    train_df = df.iloc[train_idx].copy()
    test_df = df.iloc[test_idx].copy()
    train_df.insert(0, 'dataset_index', np.asarray(train_idx, dtype=int))
    test_df.insert(0, 'dataset_index', np.asarray(test_idx, dtype=int))
    train_split_path = splits_dir / 'train_split.csv'
    test_split_path = splits_dir / 'test_split.csv'
    train_df.to_csv(train_split_path, index=False)
    test_df.to_csv(test_split_path, index=False)
    return {
        'train_indices': train_idx_path,
        'test_indices': test_idx_path,
        'train_split': train_split_path,
        'test_split': test_split_path,
    }


def write_prediction_artifact(predictions_df: pd.DataFrame, *, predictions_dir: Path, model_name: str) -> Path:
    path = predictions_dir / f'predictions_{model_name}.csv'
    predictions_df.to_csv(path, index=False)
    return path


def write_model_failure(*, metadata_dir: Path, model_name: str, status: str, error: str) -> Path:
    payload = {
        'model': model_name,
        'status': status,
        'error': error,
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S%z') or time.strftime('%Y-%m-%dT%H:%M:%S'),
    }
    path = metadata_dir / f'{model_name}_failure.json'
    path.write_text(json.dumps(payload, indent=2))
    return path


def write_research_prompt_pack(*, reports_dir: Path, results: list[dict[str, Any]], metrics: list[str], model_names: list[str], run_id: str) -> Path:
    content = build_research_prompts(results=results, metrics=metrics, model_names=model_names, run_id=run_id)
    path = reports_dir / 'research_prompt_pack.md'
    path.write_text(content)
    return path
