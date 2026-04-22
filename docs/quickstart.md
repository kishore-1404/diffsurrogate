# Quickstart

This quickstart reproduces the README minimal flow: generate example data, run a benchmark, train a production artifact, and run prediction.

1) Generate example data

```bash
python examples/generate_example_data.py
# → examples/synthetic_lookup.csv and examples/predict_inputs.csv
```

2) Run benchmark (compare all enabled surrogates)

```bash
diffsurrogate benchmark --config examples/config_fast.toml
```

3) Train production artifacts

```bash
diffsurrogate train --config examples/config_fast.toml
```

4) Predict on new points

```bash
diffsurrogate predict --config examples/config_fast.toml --models gaussian_process,pce
```

Expected outputs and locations
- `saved_models/{model}/benchmark/` — per-model benchmark artifacts
- `saved_models/{model}/production/` — production artifact bundles
- `results/` — CSV/JSON results and generated plots
