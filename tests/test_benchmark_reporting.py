"""Tests for research-grade benchmark reporting and artifact capture."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from diffsurrogate.config import (
    Config,
    DataConfig,
    EvaluationConfig,
    LoggingConfig,
    ModelsConfig,
    PersistenceConfig,
    TransformsConfig,
)
from diffsurrogate.data.splitter import stratified_split_indices_on_t
from diffsurrogate.evaluation.artifacts import (
    create_run_layout,
    write_research_prompt_pack,
    write_run_manifest,
    write_split_artifacts,
)
from diffsurrogate.evaluation.benchmark import _build_prediction_frame
from diffsurrogate.evaluation.report import write_report


class _DummyTargetScaler:
    def std_to_original(self, arr):
        return np.asarray(arr) * 0.5


def _cfg(tmp_path) -> Config:
    return Config(
        data=DataConfig(
            input_path='examples/synthetic_lookup.csv',
            predict_path='examples/predict_inputs.csv',
            input_columns=['Q2_log_center', 'W2_center', 't_center'],
            target_column='logA',
            train_fraction=0.8,
            random_seed=7,
            q2_is_prelogged=True,
        ),
        transforms=TransformsConfig(
            q2_transform='log_zscore',
            w2_transform='log_zscore',
            t_transform='log_stabilized',
            t_epsilon_frac=0.01,
            target_scaler='robust',
        ),
        models=ModelsConfig(enabled=['gaussian_process', 'neural_net']),
        evaluation=EvaluationConfig(
            metrics=['rmse', 'coverage_95'],
            output_dir=str(tmp_path / 'results'),
            save_plots=False,
        ),
        persistence=PersistenceConfig(model_dir=str(tmp_path / 'saved_models'), save_scalers=True),
        logging=LoggingConfig(level='INFO', log_file=''),
        raw={'models': {'enabled': ['gaussian_process', 'neural_net']}},
    )


def test_write_report_emits_research_files(tmp_path):
    results = [
        {
            'model': 'gaussian_process',
            'status': 'ok',
            'rmse': 0.1,
            'coverage_95': 0.93,
            'fit_time_sec': 1.2,
            'predict_time_sec': 0.02,
            'n_train': 80,
            'n_test': 20,
            'supports_uq': True,
            'artifact_size_bytes': 1234,
            'artifact_dir': '/tmp/gp',
            'predictions_path': '/tmp/gp_predictions.csv',
            'residual_mean': 0.0,
            'residual_std': 0.1,
            'abs_residual_p50': 0.05,
            'abs_residual_p90': 0.15,
            'max_abs_residual': 0.2,
        },
        {
            'model': 'neural_net',
            'status': 'failed',
            'error': 'mock failure',
            'rmse': float('nan'),
            'coverage_95': float('nan'),
            'fit_time_sec': float('nan'),
            'predict_time_sec': float('nan'),
            'n_train': 80,
            'n_test': 20,
            'supports_uq': False,
            'artifact_size_bytes': 0,
            'artifact_dir': None,
            'predictions_path': None,
            'residual_mean': float('nan'),
            'residual_std': float('nan'),
            'abs_residual_p50': float('nan'),
            'abs_residual_p90': float('nan'),
            'max_abs_residual': float('nan'),
        },
    ]

    paths = write_report(results, tmp_path, ['rmse', 'coverage_95'])

    assert paths['csv'].exists()
    assert paths['json'].exists()
    assert paths['summary_json'].exists()
    assert paths['markdown'].exists()

    summary = json.loads(paths['summary_json'].read_text())
    assert summary['metric_winners']['rmse']['model'] == 'gaussian_process'
    assert summary['failed_models'][0]['model'] == 'neural_net'

    markdown = paths['markdown'].read_text()
    assert '## Method Comparison' in markdown
    assert 'gaussian_process' in markdown


def test_split_and_manifest_artifacts_capture_indices(tmp_path):
    cfg = _cfg(tmp_path)
    X = np.column_stack([
        np.linspace(0.0, 1.0, 30),
        np.linspace(10.0, 20.0, 30),
        np.linspace(-1.0, -0.01, 30),
    ])
    df = pd.DataFrame(X, columns=cfg.data.input_columns)
    df['logA'] = np.linspace(-4.0, -1.0, 30)

    train_idx, test_idx = stratified_split_indices_on_t(X, train_fraction=0.8, random_seed=cfg.data.random_seed)
    layout = create_run_layout(cfg)
    split_paths = write_split_artifacts(df=df, train_idx=train_idx, test_idx=test_idx, splits_dir=layout['splits_dir'])
    manifest_path = write_run_manifest(
        cfg=cfg,
        df=df,
        train_idx=train_idx,
        test_idx=test_idx,
        run_id=str(layout['run_id']),
        run_dir=layout['run_dir'],
        model_names=['gaussian_process', 'neural_net'],
    )
    prompt_path = write_research_prompt_pack(
        reports_dir=layout['reports_dir'],
        results=[],
        metrics=cfg.evaluation.metrics,
        model_names=['gaussian_process', 'neural_net'],
        run_id=str(layout['run_id']),
    )

    assert split_paths['train_indices'].exists()
    assert split_paths['test_split'].exists()
    assert manifest_path.exists()
    assert prompt_path.exists()

    manifest = json.loads(manifest_path.read_text())
    assert manifest['split']['n_train'] == len(train_idx)
    assert manifest['split']['n_test'] == len(test_idx)
    assert manifest['models_enabled'] == ['gaussian_process', 'neural_net']


def test_prediction_frame_preserves_truth_predictions_and_uq_columns():
    df_test = pd.DataFrame({
        'Q2_log_center': [1.0, 2.0],
        'W2_center': [10.0, 20.0],
        't_center': [-0.2, -0.6],
        'logA': [-2.0, -1.0],
    })

    frame = _build_prediction_frame(
        df_test=df_test,
        dataset_indices=np.array([4, 9]),
        y_true_scaled=np.array([0.3, -0.1]),
        y_pred_scaled=np.array([0.1, -0.2]),
        y_true_lnA=np.array([-2.0, -1.0]),
        y_pred_lnA=np.array([-2.2, -0.8]),
        y_true_phys=np.exp(np.array([-2.0, -1.0])),
        y_pred_phys=np.exp(np.array([-2.2, -0.8])),
        y_std_scaled=np.array([0.4, 0.2]),
        target_scaler=_DummyTargetScaler(),
    )

    assert list(frame['dataset_index']) == [4, 9]
    assert 'std_ln_amplitude' in frame.columns
    assert 'inside_2sigma_interval' in frame.columns
    assert np.isclose(frame.loc[0, 'scaled_residual'], 0.2)
    assert frame['inside_2sigma_interval'].dtype.kind in {'b', 'i'}
