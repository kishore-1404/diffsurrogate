# Research Benchmarking Guide

This repository now treats benchmarking as an experiment-tracking workflow, not just a single CSV export.

What to preserve for every benchmark
- Exact `config.toml` snapshot
- Train/test indices and split CSVs
- Per-model prediction tables with residuals
- Aggregate score tables in CSV and JSON
- Human-readable benchmark summary
- Method comparison context and research prompts

What to compare across methods
- Point accuracy: RMSE and MAE
- Uncertainty quality: NLL and `coverage_95` for probabilistic models only
- Physical plausibility: `constraint_violation_rate`
- Compute cost: fit and predict time
- Tail behavior: performance around large `|t|` and diffractive dips
- Artifact reproducibility: whether the run can be reconstructed from saved files alone

Minimum reviewer-proof checklist
1. Repeat the benchmark across more than one seed.
2. Verify that train/test splits preserve the rare `|t|` regions you care about.
3. Inspect per-model prediction dumps instead of only leaderboard ranks.
4. Separate uncertainty-capable and deterministic models when discussing NLL or coverage.
5. Explain whether observed performance matches the inductive bias of each method.

Prompt templates
- Paper narrative: explain which model wins, why, and where the conclusion is fragile.
- Reviewer critique: identify missing ablations, calibration checks, or repeated-seed studies.
- Method comparison: contrast GP, Deep GP, PINN, FNO, PCE, and neural nets in terms of bias, cost, and failure modes.
- Follow-up experiments: propose robustness studies over `Q2`, `W2`, `|t|`, train fraction, and uncertainty calibration.

The benchmark command also writes a run-specific `research_prompt_pack.md` into each run bundle so interpretation can stay coupled to the saved artifacts.
