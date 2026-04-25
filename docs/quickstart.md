# Quickstart — detailed

This quickstart reproduces the README flow with extended explanations of the physics notation, the data generation, the preprocessing transforms (with math), benchmarking metrics, training persistence, and prediction post-processing.

Table of contents
- Background & notation
- 1) Generate example data (what it contains)
- 2) Preprocessing & transforms (math)
- 3) Benchmark pipeline (what happens step-by-step)
- 4) Train (production artifact contents)
- 5) Predict (how uncertainties are converted)
- 6) Reproducibility & debugging

### Background & notation

We model the exclusive diffractive scattering amplitude
$$A(Q^2, W^2, t)$$
and the repository works with its natural logarithm
$$y(Q^2,W^2,t) \equiv \ln A(Q^2,W^2,t).$$

Variables and conventions used across the codebase and examples:
- $Q^2$ — photon virtuality (GeV$^2$). Some lookup tables store $\ln Q^2$ directly.
- $W^2$ — center-of-mass energy squared (GeV$^2$).
- $t$ — Mandelstam variable (momentum transfer, GeV$^2$). Typically negative in scattering kinematics; we use $|t|$ for transforms when appropriate.

Data rows use three kinematic coordinates plus the target log-amplitude; column names are typically:
`Q2_log_center, W2_center, t_center, logA`

Physical notes:
- Diffractive dips appear at particular $|t|$ values (e.g. $|t|\approx 0.4$ GeV$^2$) and produce strong local structure in $y$.
- $y$ can vary over orders of magnitude; using the log stabilizes training for many models.

### 1) Generate example data

Run the small synthetic example generator included in `examples/`.

```bash
python examples/generate_example_data.py
# Produces:
# - examples/synthetic_lookup.csv    (gridded lookup table, columns as above)
# - examples/predict_inputs.csv      (new kinematic points for inference)
```

What the synthetic table contains
- Gridded samples in $(Q^2,W^2)$ slices across a $t$ grid to exercise the FNO mode.
- The `logA` target contains synthetic diffractive-dip structure to exercise robust scalers and stratified splitting.

Open the CSV to inspect columns and value ranges. A quick Python check:

```python
import pandas as pd
df = pd.read_csv('examples/synthetic_lookup.csv')
df.describe()
```

### 2) Preprocessing & transforms (math)

Transforms are defined in the `[transforms]` table in `config.toml`. The common transforms are:

- Q2/W2 transforms: `log_zscore` or `zscore`.
  - If `log_zscore` is selected and the column is raw $Q^2$, apply
    $$x = \log Q^2$$
    then z-score
    $$x' = \frac{x-\mu_x}{\sigma_x}$$
    where $\mu_x$ and $\sigma_x$ are computed on the training set.

- $t$ transform: `log_stabilized` applies a stabilized log followed by z-score:
  1. Choose small offset $\epsilon$ defined by
     $$\epsilon = t\_\text{epsilon\_frac} \times \mathrm{median}(|t|_{\text{train}}).$$
  2. Stable log: $$u = \log\bigl(|t| + \epsilon\bigr).$$
  3. Z-score: $$u' = \frac{u-\mu_u}{\sigma_u}.$$ 

- Target scaler: `robust` (median/IQR) or `standard` (mean/std).
  - For `robust` scaling, compute median $m$ and interquartile range $\mathrm{IQR}=Q_{3}-Q_{1}$ on training $y$ and transform
    $$y' = \frac{y-m}{\mathrm{IQR}}.$$ 
  - For `standard`, use $y'=(y-\bar y)/\sigma_y$.

Important: All transforms (means, stds, medians, IQRs, epsilons) are fitted on the training split only and saved with the artifact for inference.

### 3) Benchmark pipeline (what happens step-by-step)

The `benchmark` command executes the following high-level steps:

1. Load lookup table from `data.input_path`.
2. Apply the input transforms to form scaled features $X$ and scaled target $y$.
3. Create a stratified train/test split on deciles of $|t|$ to ensure tails and dips appear in both sets. Concretely:
   - Compute decile bin index $b = \mathrm{rank\_percentile}(|t|) // 10$ and stratify on $b$.
4. For each enabled model in `models.enabled`:
   a. Build model instance using configuration in `[models.<name>]`.
   b. Fit on $(X_{\text{train}}, y_{\text{train}})$.
   c. Predict on $X_{\text{test}}$ to obtain predicted mean $\hat y$ and (if available) predictive standard deviation $s_{\hat y}$ in log-amplitude space.
   d. Inverse-transform predictions back to physical units if desired for some metrics (e.g., amplitude-space RMSE). For log-amplitude metrics, operate directly on $y$.

Metrics computed per model
- RMSE (in log-amplitude space unless otherwise specified):
  $$\mathrm{RMSE} = \sqrt{\frac{1}{N}\sum_{i=1}^N (y_i - \hat y_i)^2}.$$ 
- Negative Log-Likelihood (NLL) for probabilistic models with Gaussian predictive distribution:
  $$\mathrm{NLL} = -\sum_{i=1}^N \log \mathcal{N}(y_i \mid \hat y_i, s_{\hat y,i}^2).$$
  For numerical stability, small floors are applied to $s_{\hat y,i}^2$.
- Coverage@95 (empirical coverage): fraction of true $y_i$ lying in $[\hat y_i \pm 1.96 s_{\hat y,i}]$.
- Unitarity violation rate: physics-specific constraint metric computed by checking predictions against domain rules (implementation-specific); the pipeline records fraction of points violating constraints.

Example benchmark command

```bash
diffsurrogate benchmark --config examples/config_fast.toml
```

Outputs
- `results/benchmark_results.csv`, `.json`, and `benchmark_summary.{json,md}` as the latest top-level summary
- `results/benchmark_runs/<run_id>/` containing the full experiment bundle for that exact run
- `results/benchmark_runs/<run_id>/predictions/predictions_<model>.csv` with truth, predictions, residuals, amplitudes, and UQ bands when available
- `results/benchmark_runs/<run_id>/splits/` with train/test indices and CSV snapshots
- Per-model directories in `saved_models/{model}/benchmark/` containing fitted artifacts and diagnostics
- ASCII leaderboard printed to console sorted by RMSE (or a chosen metric)

### 4) Train (production artifact contents)

The `train` command fits each enabled model on the full training dataset (no held-out split) and writes a production bundle. Example command:

```bash
diffsurrogate train --config examples/config_fast.toml
```

Production artifact layout (example for `gaussian_process`):

```
saved_models/gaussian_process/production/
├── sklearn_gp.joblib       # model weights / state
├── gp_mode.json            # metadata such as "exact" vs "svgp"
├── scalers.joblib          # fitted input + target scalers
└── metadata.json           # timestamp, n_train, config snapshot
```

Guidelines
- Save scalers and metadata to guarantee deterministic inference later.
- Persist the exact `config.toml` snapshot under `metadata.json` so model hyperparameters are tracked.

### 5) Predict (how uncertainties are converted)

The `predict` command loads each requested production artifact, applies the persisted transforms, and writes predictions to CSV files.

Example:

```bash
diffsurrogate predict --config examples/config_fast.toml --models gaussian_process,pce
```

Output columns typically include the input kinematics and the following prediction columns:
- `predicted_ln_amplitude` — $\hat y = \widehat{\ln A}$
- `predicted_amplitude` — $\widehat{A} = \exp(\hat y)$ (point estimate in amplitude space)
- For UQ-capable models:
  - `std_ln_amplitude` — predictive standard deviation $s_{\hat y}$ in log-amplitude space
  - `lower_2sigma`, `upper_2sigma` — amplitude-space intervals computed by exponentiating the log-space bounds:
    $$\text{lower} = \exp\bigl(\hat y - 2 s_{\hat y}\bigr),\qquad \text{upper} = \exp\bigl(\hat y + 2 s_{\hat y}\bigr).$$

Notes on converting uncertainties
- The exact mapping from a Gaussian predictive distribution in log-space to amplitude-space is log-normal. If $Y\sim\mathcal{N}(\mu,\sigma^2)$ then $A=\exp(Y)$ has
  mean $\mathbb{E}[A]=\exp\left(\mu+\tfrac{1}{2}\sigma^2\right)$ and median $\exp(\mu)$. The code reports the exponentiated central estimate and simple $\pm 2\sigma$ amplitude intervals by default; consult `evaluation` utilities if you need log-normal-corrected means.

### 6) Reproducibility & debugging

- Seeds: set `data.random_seed` in `config.toml` to control numpy, torch, and random module seeds used by splitter and model code. Exact determinism may vary across backends (GPU nondeterminism, variational sampling in deep GP).
- Missing optional packages: the CLI warns and skips models whose dependencies are unavailable.
- Common issues and fixes:
  - "Model training fails due to ill-conditioned covariance": try tightening GP length-scale bounds in the GP config or increase noise/regularization.
  - "FNO warns about ungridded t": ensure your lookup table is gridded on `t` for FNO-1d mode; otherwise FNO falls back to an MLP.

### Appendix: example `config.toml` snippet

```toml
[data]
input_path = "examples/synthetic_lookup.csv"
predict_path = "examples/predict_inputs.csv"
input_columns = ["Q2_log_center","W2_center","t_center"]
target_column = "logA"
random_seed = 42

[transforms]
q2_transform = "log_zscore"
w2_transform = "log_zscore"
t_transform = "log_stabilized"
t_epsilon_frac = 0.01
target_scaler = "robust"

[models]
enabled = ["neural_net","gaussian_process","pce"]
```

This completes the quickstart with extended explanations. For more detail on model internals, see [docs/models.md](models.md) and for transform math consult [docs/data.md](data.md).
