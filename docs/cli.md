# CLI Reference

`diffsurrogate` provides three primary CLI subcommands implemented in `diffsurrogate/cli/`:

- `benchmark` — Evaluate and compare enabled surrogates on a held-out split
- `train` — Fit models on the full dataset and save production artifacts
- `predict` — Load saved artifacts and run inference on new points

Common options (see `--help` for exact flags):

- `--config` (`-c`) — Path to `config.toml` (required)
- `--models` — Comma-separated list of models to restrict execution
- `--output` — Override output directory

Examples

```bash
diffsurrogate benchmark --config examples/config_fast.toml
diffsurrogate train --config examples/config_fast.toml --models neural_net
diffsurrogate predict --config examples/config_fast.toml --models gaussian_process,pce
```

Implementation notes
- CLI entrypoints are in [diffsurrogate/cli/main.py](diffsurrogate/cli/main.py). Individual command logic is implemented in the respective modules: [diffsurrogate/cli/benchmark_cmd.py](diffsurrogate/cli/benchmark_cmd.py), [diffsurrogate/cli/train_cmd.py](diffsurrogate/cli/train_cmd.py), and [diffsurrogate/cli/predict_cmd.py](diffsurrogate/cli/predict_cmd.py).
