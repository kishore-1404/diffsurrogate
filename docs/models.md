## Models — deep reference

This document describes the surrogate paradigms implemented in `diffsurrogate`, their mathematical foundations, typical hyperparameters, expected runtime/behaviour, how uncertainty is produced and interpreted, persistence format, and known pitfalls. All models implement the abstract interface `SurrogateModel` defined in [diffsurrogate/models/base.py](diffsurrogate/models/base.py) with core methods: `name()`, `supports_uq()`, `fit(X,y)`, `predict(X)`, `save(path)`, and `load(path)`.

Table of contents
- Common interface and conventions
- Data shapes and notation
- Model-specific sections:
	- `neural_net`
	- `gaussian_process` (exact + sparse)
	- `deep_gp`
	- `pinn`
	- `fno`
	- `pce`
- Persistence and reproducibility
- Hyperparameter examples (config snippets)
- Testing, validation and performance tips
- Limitations and open issues

### Common interface and conventions

- Inputs: features are arranged as numeric arrays of shape $(N, D)$ where columns follow `config.data.input_columns` order: $(Q^2\_col, W^2\_col, t\_col)$ after transforms.
- Targets: single-column $y=\ln A$ scaled according to `transforms.target_scaler`.
- All models are expected to accept scaled-space arrays for `fit()` and `predict()`. The CLI and pipeline handle transforms automatically — model authors should assume pre-scaled inputs.

Notation:
- $X \in \mathbb{R}^{N\times D}$ — feature matrix, $D=3$ here (Q2, W2, t) unless additional engineered features added.
- $y \in \mathbb{R}^{N}$ — target log-amplitude vector.

### `neural_net` (ResNet-style deterministic)

Overview
- Implements a feedforward residual MLP/ResNet backbone suitable as a fast deterministic surrogate. Designed as a baseline for speed and capacity; does not provide UQ by default.

Architecture
- Typical block: fully-connected layer → activation (ELU/ReLU) → optional batch-norm → residual skip. A small head projects to a scalar output.

Loss
- Mean-squared error on scaled $y$ during training:
	$$\mathcal{L}_{\text{data}} = \frac{1}{N}\sum_{i=1}^N (y_i - \hat y_i)^2.$$ 

Hyperparameters (config keys)
- `hidden_sizes` — list of widths per hidden layer
- `n_blocks` — number of residual blocks
- `activation` — e.g., `elu`, `relu`
- `dropout` — dropout probability (optional)
- `lr` — learning rate
- `batch_size`, `epochs`, `optimizer` (adam/sgd)

Training tips
- Use `robust` target scaling for diffractive dips.
- Monitor validation RMSE and early-stop using patience (5–20 epochs) to avoid overfitting on narrow dips.

Prediction & UQ
- Deterministic by default. For approximate UQ, enable Monte-Carlo Dropout at inference (keep dropout active and run multiple forward passes), which approximates Bayesian model averaging:
	$$\hat y = \frac{1}{M}\sum_{m=1}^M f_{\theta}^{(m)}(x), \quad s_{\hat y}^2 = \frac{1}{M-1}\sum_{m=1}^M (f^{(m)}-\hat y)^2.$$ 

Persistence
- Saved via PyTorch state dict or exported weights; `scalers.joblib` saved separately by pipeline.

When to use
- When speed is primary and UQ is not required or can be approximated with MC Dropout.

### `gaussian_process` (exact & sparse SVGP)

Overview
- Two GP modes are supported: exact Gaussian Process regression (suitable for smaller datasets) and sparse variational GP (SVGP) using inducing points for scalability.

Model basics (exact GP)
- Prior: $f(\cdot) \sim \mathcal{GP}(0, k(\cdot,\cdot))$ with kernel $k$. The common choice here is Matérn-5/2 with Automatic Relevance Determination (ARD) lengthscales.
- Given training $(X,y)$ with Gaussian observation noise $\sigma_n^2$, the predictive distribution at $x_*$ is Gaussian with mean and variance:
	$$\mu_* = k_*^T (K + \sigma_n^2 I)^{-1} y, \qquad \sigma_*^2 = k_{**} - k_*^T (K + \sigma_n^2 I)^{-1} k_*,$$
	where $K$ is the $N\times N$ kernel matrix, $k_* = k(X,x_*), k_{**}=k(x_*,x_*).$

Numerical notes
- Exact GP scales as $\mathcal{O}(N^3)$ in time and $\mathcal{O}(N^2)$ in memory; suitable for $N\lesssim$ a few thousand. For larger datasets use SVGP.

SVGP (sparse variational GP)
- Uses $M\ll N$ inducing points $Z$ and variational parameters to approximate the posterior. Training maximizes an evidence lower bound (ELBO) using stochastic optimization.

Hyperparameters (config keys)
- `kernel` — `mattern52_ard` (default) or alternatives
- `lengthscale_bounds` — tightened bounds recommended: e.g. `[1e-2, 1e2]` because inputs are z-scored
- `noise` — observation noise prior/init
- `svgp.num_inducing` — number of inducing points for SVGP
- optimization: `lr`, `maxiter`, `optimizer` (L-BFGS-B used for exact GP by default)

Uncertainty and NLL
- GP predictive variance $\sigma_*^2$ is principled and can be used to compute NLL and coverage directly.
- NLL for Gaussian predictive: see quickstart NLL formula.

Practical tips
- Standardize inputs before GP: z-scored inputs reduce pathological lengthscale behavior.
- Tighten `lengthscale_bounds` after z-scoring (e.g., $(10^{-2},10^{2})$) to prevent lengthscale collapse to extreme boundaries.

Persistence
- Exact GP (scikit-learn style) saved via `joblib.dump` of the fitted `GaussianProcessRegressor`.
- SVGP persists model state (PyTorch state dict or gpytorch equivalent) and a `svgp_meta.json` describing inducing point locations and training hyperparams.

When to use
- When principled UQ is required and dataset sizes permit exact or sparse GP.

### `deep_gp` (hierarchical GP via variational inference)

Overview
- Deep Gaussian Processes stack GPs in a hierarchical fashion, allowing for non-Gaussian function families and higher expressive power than shallow GPs. Implemented with doubly-stochastic variational inference (DSVI) or similar.

Model sketch
- A two-layer deep GP with latent $h$:
	$$h(x) \sim \mathcal{GP}_1(0,k_1),\qquad f(h) \sim \mathcal{GP}_2(0,k_2)$$
- Variational inference approximates the posterior for layer outputs; predictions are obtained by Monte Carlo sampling through the layers.

Training objective
- Maximize ELBO; rely on mini-batching and Monte Carlo samples for gradients.

Uncertainty
- Predictive uncertainty combines aleatoric and epistemic components approximated via MC sampling. Because sampling occurs through variational layers, repeated fits may produce stochastic differences; saved state + fixed RNG reduces but may not eliminate variation.

Hyperparameters
- `n_layers`, `n_inducing_per_layer`, `n_samples_predict`, `lr`, `epochs`.

Persistence & notes
- Persist model state dict and `metadata.json` with `n_samples_predict` used during eval. The persistence test tolerances in `tests/test_models.py` are looser for DeepGP due to sampling variability.

### `pinn` (Physics-Informed Neural Network)

Overview
- PINNs augment data loss with physics constraints derived from known equations (e.g., BK evolution, unitarity constraints). This encourages physically-plausible extrapolations and reduces overfitting to noisy labels.

Loss structure
- Typical composite loss:
	$$\mathcal{L} = \lambda_{\text{data}} \mathcal{L}_{\text{data}} + \lambda_{\text{phys}} \mathcal{L}_{\text{phys}} + \lambda_{\text{reg}} \mathcal{L}_{\text{reg}}.$$ 
- $\mathcal{L}_{\text{data}}$ — MSE on observed labels.
- $\mathcal{L}_{\text{phys}}$ — residual of the physical constraint (for example, a PDE residual or unitarity penalty). If $\mathcal{P}[A]=0$ is a physics constraint, then
	$$\mathcal{L}_{\text{phys}} = \frac{1}{N_p} \sum_{j=1}^{N_p} \left|\mathcal{P}[\widehat{A}(x_j)]\right|^2$$
	evaluated at collocation points $x_j$ (could be the same as training inputs or a separate grid).

Physics used in this project
- The PINN here is designed to embed BK evolution and unitarity constraints as soft penalties — refer to `diffsurrogate/models/pinn.py` for exact terms used.

Hyperparameters
- `lambda_phys`, `lambda_reg`, `n_collocation`, `phys_equation_config` (if available), network architecture similar to `neural_net`.

Training notes
- Balance data and physics loss carefully; scale terms so gradients are comparable. Use learning rate scheduling when training with mixed losses.

UQ
- PINNs may be augmented with Bayesian NN techniques or ensembles to obtain uncertainty estimates; not provided out-of-the-box.

### `fno` (Fourier Neural Operator — 1d over t)

Overview
- FNO is applied along the $t$ axis for gridded $(Q^2,W^2)$ slices. It learns integral operators via Fourier-domain multipliers and is powerful on gridded function regression tasks.

Detection & input requirements
- To use the FNO branch, the loader must detect that the lookup table is gridded: for each $(Q^2,W^2)$ slice, the set of `t_center` values must be identical and ordered. If the dataset is not gridded, the pipeline falls back to a ResNet-style MLP.

Mathematical sketch
- Represent the function over $t$ for fixed $(Q^2,W^2)$ as $u(t)$ sampled on grid points $t_j, j=1\dots L$. FNO learns an operator $\mathcal{G}$ with spectral layers:
	$$u_{\ell+1}(t) = \sigma\Bigl(\mathcal{F}^{-1} \bigl( P_{\ell}(k) \cdot \mathcal{F}(u_{\ell})(k) \bigr) + W_{\ell} u_{\ell}(t) \Bigr)$$
	where $\mathcal{F}$ is discrete Fourier transform across the grid, $P_{\ell}(k)$ are learned multipliers, and $W_{\ell}$ are local linear operators.

Hyperparameters
- `modes` — number of Fourier modes kept
- `width` — channel width
- `n_layers`, `lr`, `epochs`.

When to use
- When data is gridded in $t$ and the underlying function is smooth/structured across $t$ (FNO excels at learning global operators).

### `pce` (Polynomial Chaos Expansion)

Overview
- Polynomial Chaos Expansion (PCE) represents the target as a spectral expansion in orthogonal polynomials (Hermite polynomials for Gaussian-like inputs). PCE is analytical and can provide closed-form UQ via coefficient covariances when built via regression.

Formulation
- Represent $y(x)$ as
	$$y(x) \approx \sum_{\alpha \in \mathcal{A}} c_{\alpha} \Psi_{\alpha}(x)$$
	where $\Psi_{\alpha}$ are multivariate orthogonal polynomial basis functions (multi-indices $\alpha$) and $c_{\alpha}$ are coefficients estimated by regression (ordinary least squares or regularized variants).

Hyperparameters
- `max_order` — highest polynomial order included
- `basis` — Hermite (default) for near-Gaussian inputs
- `reg` — optional regularization (ridge)

UQ
- PCE yields a coefficient covariance matrix under linear regression assumptions; propagate to predictive variance analytically.

When to use
- When the response is well-approximated by a low-order polynomial expansion in the input features and when interpretability and analytic UQ are desired.

### Persistence, saving & loading

General guidelines
- Always save scalers (input + target) along with model weights/state. The pipeline stores these in `scalers.joblib` so inference uses identical transforms.
- Save a `metadata.json` containing:
	- training timestamp
	- `n_train`
	- a snapshot of the `config.toml` used
	- exact hyperparameters

Model-specific persistence
- `neural_net`: PyTorch `state_dict()` and a small JSON describing architecture
- `gaussian_process`: `sklearn` GP saved via `joblib` or SVGP state dicts + `svgp_meta.json`
- `deep_gp`: PyTorch/gpytorch state dict + `metadata.json`
- `pce`: coefficients and basis information saved via `joblib` or `numpy.save`

### Hyperparameter examples (config snippets)

Neural net (in `config.toml`):
```toml
[models.neural_net]
hidden_sizes = [128,128,64]
n_blocks = 3
activation = "elu"
dropout = 0.1
lr = 1e-3
batch_size = 64
epochs = 200
```

Gaussian Process (exact) example:
```toml
[models.gaussian_process]
mode = "exact" # or "svgp"
kernel = "mattern52_ard"
lengthscale_bounds = [1e-2, 1e2]
noise = 1e-6
optimizer = "L-BFGS-B"
```

SVGP example:
```toml
[models.gaussian_process]
mode = "svgp"
num_inducing = 128
lr = 1e-2
epochs = 500
```

PCE example:
```toml
[models.pce]
max_order = 4
basis = "hermite"
reg = 1e-6
```

### Testing, validation and performance tips

- Unit tests: add smoke tests in `tests/test_models.py` following the repo pattern. Each model should train a few iterations on synthetic data and verify that `save`/`load` round-trips produce similar predictions within tolerance.
- Performance:
	- Use vectorized batching for predictions.
	- For GPs and DeepGPs, prefer SVGP for large N to avoid cubic scaling.

### Limitations and pitfalls

- FNO requires gridded `t`; feeding irregular $t$ will either fail or degrade performance.
- Exact GP is not feasible for large training sets due to cubic complexity.
- DeepGPs involve stochastic variational training and may produce non-deterministic results; increase `n_samples_predict` and fix random seeds where reproducibility is required.
- PINNs require careful weighting of physics vs data loss; naive weighting can harm fit.

### References and further reading

- R. G. G. L. Gibson et al., "Fourier Neural Operators" — for FNO background.
- Rasmussen & Williams, "Gaussian Processes for Machine Learning" — for GP theory.
- Raissi, Perdikaris, Karniadakis, "Physics-informed neural networks" — for PINN methodology.

If you want, I can now create separate per-model markdown pages under `docs/models/` (e.g. `docs/models/neural_net.md`, `docs/models/gaussian_process.md`, etc.) with runnable config examples, visual diagnostic examples, and more extended math derivations. Proceed? 

