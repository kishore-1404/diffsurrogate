# Persistence & Saved Artifacts

Saved model bundles are organized under `saved_models/{model}/{mode}/` where `{mode}` is `benchmark` or `production`.

Typical production bundle
```
saved_models/gaussian_process/production/
├── sklearn_gp.joblib
├── gp_mode.json
├── scalers.joblib
└── metadata.json
```

Benchmark-time persistence
- During benchmarking, each successful model is still saved under `saved_models/{model}/benchmark/`.
- `metadata.json` now includes the benchmark `run_id` in the config snapshot so model artifacts can be traced back to a specific benchmark bundle.
- The heavy research artifacts live under `results/benchmark_runs/<run_id>/`, not inside `saved_models/`.

Recommended usage
- Use `saved_models/{model}/production/` for deployment or downstream inference.
- Use `results/benchmark_runs/<run_id>/` for scientific record-keeping, comparisons, and paper figures.
- Keep `metadata.json`, `scalers.joblib`, and the benchmark run manifest together if you need exact reproducibility later.
