# Configuration Reference

All runtime behaviour is driven by `config.toml` at the repository root. Use the example config files in `examples/` as starting points.

Top-level sections
- `[data]` — input paths, column names, splitting, seeds
- `[transforms]` — input and target scalers and t-stabilization settings
- `[models]` — `enabled` list and per-model hyperparameters
- `[evaluation]` — metrics, plots, benchmark artifact capture, and reporting
- `[persistence]` — model bundle directories
- `[logging]` — stderr/file logging

Important keys
- `data.input_path` — path to lookup table (CSV/HDF5/.npy)
- `data.predict_path` — path with new points for prediction
- `data.train_fraction` — held-out fraction for benchmark mode
- `transforms.target_scaler` — `robust` is recommended for diffractive dips
- `models.enabled` — list of model names to run

Evaluation-specific keys
- `evaluation.metrics` — metrics to compute for each model
- `evaluation.output_dir` — root location for summary outputs and benchmark run bundles
- `evaluation.save_plots` — save residual/UQ/t-spectrum figures
- `evaluation.save_predictions` — save full per-model prediction CSVs
- `evaluation.save_split_data` — save train/test indices and split CSVs
- `evaluation.save_run_manifest` — save run metadata and config snapshot JSON
- `evaluation.write_markdown_report` — emit a readable Markdown benchmark summary
- `evaluation.benchmark_runs_dirname` — subdirectory used for timestamped runs

When adding a new model, add a `[models.new_model]` table and include the name in `models.enabled`.
