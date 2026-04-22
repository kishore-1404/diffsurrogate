# diffsurrogate

Surrogate modeling for diffractive scattering observables in QCD.

Six surrogate paradigms evaluated head-to-head against precomputed lookup
tables of the exclusive diffractive scattering amplitude $\ln A(Q^2, W^2, t)$:

- **Neural Network / ResNet** — fast deterministic baseline
- **Exact & Sparse Gaussian Process** — Bayesian, exact UQ, Matérn-5/2 ARD
- **Physics-Informed Neural Network (PINN)** — embeds BK evolution + unitarity
- **Fourier Neural Operator (FNO-1d)** — along the $t$-axis for gridded data
- **Deep Gaussian Process** — hierarchical, DSVI
- **Polynomial Chaos Expansion** — analytical, spectral, Hermite basis

All six share a single abstract `SurrogateModel` interface, a single
`config.toml`, and three CLI modes: benchmark, train, and predict.

---

## Installation

```bash
git clone <repo>
cd diffsurrogate
pip install -e .[all]        # core + torch + gpytorch + chaospy + h5py
```

Install subsets if you don't need every backend:

```bash
pip install -e .[torch]          # NN, PINN, FNO
pip install -e .[gpytorch]       # Deep GP, sparse GP
pip install -e .[chaospy]        # PCE
pip install -e .[test]           # pytest
```

Missing an optional dependency is **never fatal**: the affected model is
skipped with a warning and the rest of the pipeline continues.

### Requirements

- Python ≥ 3.11
- Core (always): numpy, pandas, scipy, scikit-learn, joblib, matplotlib

### Verified versions

| Package | Tested against |
|---|---|
| torch | 2.11 (CPU works) |
| gpytorch | 1.15 |
| chaospy | 4.3 |
| scikit-learn | 1.3+ |

---

## Quick start

### 1. Generate example data

```bash
python examples/generate_example_data.py
# → examples/synthetic_lookup.csv    (960 rows, gridded on t)
# → examples/predict_inputs.csv      (20 new kinematic points)
```

The synthetic data mimics Sartre lookup tables: columns are
`Q2_log_center, W2_center, t_center, logA`, with a diffractive-dip structure
at $|t| \approx 0.4$ GeV².

### 2. Benchmark — compare all surrogates

```bash
diffsurrogate benchmark --config examples/config_fast.toml
```

Pipeline:
1. Loads the lookup table
2. Applies physics-informed scalers (Q²/W² log-z-scored, t log-stabilized,
   target robust-scaled)
3. Stratified split on |t| deciles (so diffractive dips appear in both
   train and test)
4. Fits each enabled model on the training split, predicts on test
5. Computes RMSE, NLL, coverage-95, and unitarity violation rate
6. Saves artifacts to `saved_models/{name}/benchmark/`
7. Writes `results/benchmark_results.{csv,json}` plus plots
8. Prints an ASCII leaderboard ranked by RMSE

Expected output (approximate, on the synthetic example data):

```
+------------------+--------+---------+-------------+---------------------------+--------------+
| model            | rmse   | nll     | coverage_95 | constraint_violation_rate | fit_time_sec |
+------------------+--------+---------+-------------+---------------------------+--------------+
| gaussian_process | 0.0055 | -3.7740 | 0.9211      | 0.000e+00                 | 34.37        |
| fno              | 0.3194 | nan     | nan         | 0.000e+00                 | 4.39         |
| pce              | 0.3205 | 0.2831  | 0.9421      | 0.000e+00                 | 0.68         |
| neural_net       | 0.3528 | nan     | nan         | 0.000e+00                 | 6.10         |
| pinn             | 0.3668 | nan     | nan         | 0.000e+00                 | 3.86         |
| deep_gp          | 0.3730 | 0.7365  | 0.9895      | 0.000e+00                 | 8.97         |
+------------------+--------+---------+-------------+---------------------------+--------------+
```

GP dominates on low-noise data (as the paper predicts). NLL / coverage-95
are NaN for models that don't expose uncertainty.

### 3. Train — produce production artifacts

```bash
diffsurrogate train --config examples/config_fast.toml
```

Fits scalers on the **full** dataset (no held-out split) and saves every
enabled model to `saved_models/{name}/production/`. The saved bundle
contains:

```
saved_models/gaussian_process/production/
├── sklearn_gp.joblib       # or svgp_state.pt + svgp_meta.json for SVGP
├── gp_mode.json            # "exact" vs "svgp"
├── scalers.joblib          # fitted input + target scalers
└── metadata.json           # timestamp, n_train, config snapshot
```

### 4. Predict — inference on new points

```bash
diffsurrogate predict --config examples/config_fast.toml
diffsurrogate predict --config examples/config_fast.toml --models gaussian_process,pce
```

Loads each enabled production artifact, applies the **saved** scalers (never
refitted), predicts, inverse-transforms to physical amplitude units, and
writes `results/predictions_{model}.csv` with columns:

| Column | Always | UQ models only |
|---|---|---|
| `Q2_log_center, W2_center, t_center` | ✓ | ✓ |
| `predicted_ln_amplitude` | ✓ |  |
| `predicted_amplitude` (= exp of the above) | ✓ |  |
| `std_ln_amplitude` |  | ✓ |
| `lower_2sigma`, `upper_2sigma` (in amplitude space) |  | ✓ |

---

## Configuration reference

Everything is driven by `config.toml`. No value used at runtime is
hardcoded in Python — if you find yourself wanting a knob, add it here
and wire it through.

### `[data]`

| Key | Meaning |
|---|---|
| `input_path` | Path to the lookup table (`.csv`, `.h5`, `.hdf5`, or `.npy` with a structured dtype) |
| `predict_path` | CSV of new kinematic points for `predict` mode |
| `input_columns` | The three column names, in `(Q², W², t)` order |
| `q2_is_prelogged` | `true` if the Q² column already contains $\ln Q^2$ (Sartre default); `false` for raw $Q^2$ |
| `target_column` | Target column name (e.g. `"logA"`) |
| `train_fraction` | Train/test split fraction — benchmark mode only |
| `random_seed` | Master seed (threaded to numpy, torch, python random, and the stratified splitter) |

### `[transforms]`

| Key | Options |
|---|---|
| `q2_transform`, `w2_transform` | `"log_zscore"` \| `"zscore"` |
| `t_transform` | `"log_stabilized"` — applies $\log(\|t\| + \epsilon)$ then z-scores |
| `t_epsilon_frac` | $\epsilon = \texttt{t\_epsilon\_frac} \times \text{median}(\|t\|)$ on training data only |
| `target_scaler` | `"robust"` (median/IQR — recommended for diffractive dips) \| `"standard"` |

### `[models]`

```toml
[models]
enabled = ["neural_net", "gaussian_process", "pinn", "fno", "deep_gp", "pce"]
```

Removing a name from `enabled` skips that model (its sub-table is then
ignored). Each sub-table holds the hyperparameters for its model — see
the top-level `config.toml` for a fully-documented template.

### `[evaluation]`, `[persistence]`, `[logging]`

See the inline comments in `config.toml`.

---

## Extending with a new surrogate

1. Subclass `SurrogateModel` in `diffsurrogate/models/your_model.py`:

   ```python
   from pathlib import Path
   import numpy as np
   from diffsurrogate.models.base import SurrogateModel

   class YourModelSurrogate(SurrogateModel):
       def __init__(self, cfg): self.cfg = cfg; self._m = None

       def name(self) -> str: return "your_model"
       def supports_uq(self) -> bool: return False

       def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
           # X and y are already in scaled space
           self._m = ...

       def predict(self, X: np.ndarray):
           return self._m.predict(X).ravel(), None   # (mean, std-or-None)

       def save(self, path: Path) -> None:
           path.mkdir(parents=True, exist_ok=True)
           # save weights / state to `path/`

       def load(self, path: Path) -> None:
           # restore from `path/`
           ...
   ```

2. Add a config dataclass in `diffsurrogate/config.py` alongside
   `NeuralNetConfig` et al., and register it in `ModelsConfig`.

3. Register the name → class mapping in
   `diffsurrogate/persistence/registry.py`:

   ```python
   def _your_model():
       from diffsurrogate.models.your_model import YourModelSurrogate
       from diffsurrogate.config import YourModelConfig
       return YourModelSurrogate, YourModelConfig

   MODEL_REGISTRY["your_model"] = _your_model
   ```

4. Add a builder branch in `diffsurrogate/models/__init__.py`'s
   `build_model`.

5. Add a `[models.your_model]` sub-table to `config.toml` and add
   `"your_model"` to `enabled`.

6. Write a smoke test in `tests/test_models.py` — just add the name plus
   a `_default_cfg` branch, the parametrized fixtures handle the rest.

---

## Running the tests

```bash
pip install -e .[test,all]
pytest tests/ -v
```

Expected output: **78 passed** (21 transforms + 20 metrics + 20 models +
17 persistence).

Tests skip gracefully when an optional dependency is missing — e.g. on a
machine without `gpytorch`, Deep GP tests are skipped rather than failing.

---

## Design notes

**Why robust (median/IQR) target scaling?** Diffractive dips produce
severe negative outliers in $\ln A$. Mean/std scaling gets pulled by
them; median/IQR doesn't.

**Why stratified split on |t| deciles?** A naive random split can put all
the rare deep-dip points into one side. Stratifying guarantees the tails
appear in both train and test, which is what you care about.

**Why `q2_is_prelogged`?** Sartre lookup tables store $\ln Q^2$ directly
(as `Q2_log_center`); a user with raw $Q^2$ should flip the flag so the
scaler applies `log()` first.

**GP length-scale bounds.** Because inputs are z-scored to roughly $[-3,
3]$, we tighten sklearn's default `length_scale_bounds` from $(10^{-5},
10^{5})$ to $(10^{-2}, 10^{2})$. Without this, the L-BFGS-B optimizer
occasionally clamps a length scale to the boundary and produces a
constant-output GP. See `models/gaussian_process.py`.

**FNO on tabular data.** Paper defines FNO over spatial density fields.
For tabular $(Q^2, W^2, t)$ triples we:
- Detect whether the data is gridded on $t$ across shared $(Q^2, W^2)$
  slices.
- If so, train an FNO-1d along $t$ (the axis where diffractive
  structure lives).
- Otherwise warn and fall back to a ResNet-style MLP.

Both modes are visible in `metadata.json` after training.

**Deep GP determinism.** `DeepGP.predict()` draws `n_samples` MC samples
through the variational stack and averages. Identical weights give
non-identical forward passes. The persistence round-trip test uses a
loose tolerance for this model (≈ 2.5e-2 on the synthetic data).

---

## License

MIT.
