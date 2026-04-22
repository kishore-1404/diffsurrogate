# Gaussian Process surrogates (Exact & SVGP)

This page documents the `gaussian_process` surrogate: kernels, hyperparameters, numerical stability, and persistence.

Introduction
- Gaussian Processes (GPs) provide a nonparametric Bayesian approach to regression with closed-form predictive distributions for Gaussian likelihoods. They are an excellent choice when principled uncertainty quantification (UQ) is required.

Model equations (exact GP)
- Prior: $f(\cdot) \sim \mathcal{GP}(0,k(\cdot,\cdot))$.
- Given observations $(X,y)$ with Gaussian noise variance $\sigma_n^2$, the posterior predictive distribution at test point $x_*$ is Gaussian with
  $$\mu_* = k_*^T (K + \sigma_n^2 I)^{-1} y, \qquad \sigma_*^2 = k_{**} - k_*^T (K + \sigma_n^2 I)^{-1} k_*.$$

Kernel choices
- Matérn-5/2 ARD (default):
  $$k(r) = \sigma_f^2 \left(1 + \sqrt{5} r + \tfrac{5}{3} r^2\right)\exp(-\sqrt{5} r),$$
  where $r = \sqrt{\sum_d \left(\frac{x_d-x'_d}{\ell_d}\right)^2 }$ and $\ell_d$ are lengthscales (ARD).
- Squared exponential (RBF) is available but sometimes over-smooths diffractive structure.

Hyperparameters and bounds
- `lengthscale_bounds` — Because inputs are z-scored, use tightened bounds such as `[1e-2, 1e2]` to avoid optimizers pushing bounds to extreme values.
- `sigma_f` — signal variance
- `sigma_n` — observation noise; if too small, numerical inversion may be unstable.

Numerical stability tips
- Add jitter (small diagonal term) when inverting $K$ for numerical stability (e.g., jitter = 1e-8 to 1e-6).
- Use Cholesky decomposition for solving linear systems; catch LinAlgError and increase jitter if decomposition fails.

Scalability: SVGP
- Exact GP scales poorly as $\mathcal{O}(N^3)$. Use Sparse Variational GP (SVGP) for large datasets with $M$ inducing points; complexity reduces to $\mathcal{O}(NM^2)$ per epoch for naive implementations and can be minibatched.

Training & inference
- Exact GP: optimize marginal likelihood (type-II ML) with L-BFGS-B or similar quasi-Newton optimiser; converges quickly for moderate N.
- SVGP: optimize ELBO using Adam with minibatching; requires sensible initialization of inducing points (kmeans or subset of data).

Uncertainty & Metrics
- Use predictive variance $\sigma_*^2$ for NLL and coverage metrics.
- For log-amplitude predictions, NLL formula is
  $$\mathrm{NLL} = -\sum_i \log \mathcal{N}(y_i\mid \mu_i, \sigma_i^2).$$

Persistence
- Exact GP: saved with `joblib.dump(gpr)` which includes kernel hyperparameters.
- SVGP: save model state dict and `svgp_meta.json` containing inducing points and ELBO hyperparameters.

Configuration examples
```toml
[models.gaussian_process]
mode = "exact"
kernel = "mattern52_ard"
lengthscale_bounds = [1e-2, 1e2]
noise = 1e-6
optimizer = "L-BFGS-B"

# For SVGP
[models.gaussian_process]
mode = "svgp"
num_inducing = 128
lr = 1e-2
epochs = 500
```

Common failure modes
- Lengthscale collapses to tiny values: increase lower bound or add input regularization.
- Cholesky fails: increase jitter.

Practical notes
- Standardize inputs and use robust target scaling to improve GP behaviour on heavy-tailed targets.

Example config for quick experiments: [examples/config_gaussian_process.toml](../examples/config_gaussian_process.toml)
