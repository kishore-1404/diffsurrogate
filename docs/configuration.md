# Configuration Reference

All runtime behaviour is driven by `config.toml` at the repository root. Use the examples config files in `examples/` as starting points.

Top-level sections you will see in templates:

- `[data]` — input paths, column names, splitting, seeds
- `[transforms]` — input and target scalers and t-stabilization settings
- `[models]` — `enabled` list and per-model hyperparameters
- `[evaluation]` — metric choices, plotting options
- `[persistence]` — output directories and naming

Important keys

- `data.input_path` — path to lookup table (CSV/HDF5/.npy)
- `data.predict_path` — path with new points for prediction
- `transforms.target_scaler` — `robust` recommended for diffractive dips
- `models.enabled` — list of model names to run

When adding a new model, add a `[models.new_model]` table and a name in `enabled`.
