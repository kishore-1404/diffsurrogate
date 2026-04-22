# Evaluation & Metrics

Evaluation metrics and plotting live under `diffsurrogate/evaluation/`.

Metrics computed by the benchmark pipeline include:
- RMSE — root-mean-square error against lookup table
- NLL — negative log-likelihood (UQ-capable models only)
- coverage_95 — empirical coverage of the 95% interval (UQ models)
- unitarity violation rate — proportion of predictions breaching physics constraints

Benchmark pipeline summary
1. Train models on the training split
2. Predict on the test split
3. Compute metrics and generate plots
4. Save results in `results/` and per-model artifacts in `saved_models/`

Plots
- Leaderboard table (ASCII) printed to console
- Diagnostic plots saved to `results/plots/`
