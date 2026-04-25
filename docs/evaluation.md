# Evaluation & Benchmarking

Evaluation code lives under `diffsurrogate/evaluation/`.

Core metrics
- RMSE — point-prediction error in scaled target space
- MAE — absolute error in scaled target space
- NLL — Gaussian negative log-likelihood for UQ-capable models
- `coverage_95` — empirical fraction of test points inside the `±2σ` interval
- `constraint_violation_rate` — fraction of physical-space predictions outside the admissible range

Research-grade benchmark bundle
- Every benchmark run now creates `results/benchmark_runs/<run_id>/`.
- `metadata/run_manifest.json` stores the config snapshot, enabled methods, dataset summary, and split summary.
- `splits/` stores train/test indices plus CSV snapshots of both splits.
- `predictions/` stores one CSV per model with ground truth, predictions, residuals, amplitudes, and uncertainty intervals when available.
- `reports/` stores CSV, JSON, Markdown, and summary JSON reports for the run.
- `reports/research_prompt_pack.md` provides paper-writing and reviewer-style prompts for interpreting the benchmark scientifically.
- `plots/` stores residual, UQ, and t-spectrum figures when plotting is enabled.

Comparison logic
- Aggregate reports still preserve the configured metric list.
- The benchmark summary also records residual diagnostics such as median absolute residual, 90th percentile absolute residual, and maximum absolute residual.
- Markdown summaries include a method comparison table covering probabilistic support, physics-informed structure, grid requirements, and compute profile.

Recommended research workflow
1. Run `diffsurrogate benchmark --config <config.toml>`.
2. Treat `benchmark_results.csv` as the quick glance only.
3. Use the timestamped run bundle for any claim you intend to cite in a report or paper.
4. Inspect the per-model prediction CSVs before making conclusions about behavior around diffractive dips or large `|t|`.
5. Re-run across multiple seeds or train fractions before making a strong ranking claim.
