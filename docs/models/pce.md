# Polynomial Chaos Expansion (PCE)

PCE represents the target as a spectral expansion in orthogonal polynomials. It is analytical, interpretable, and can produce closed-form uncertainty estimates under linear regression assumptions.

Mathematical formulation
- Represent the model as
  $$y(x) \approx \sum_{\alpha\in\mathcal{A}} c_{\alpha} \Psi_{\alpha}(x),$$
  where $\Psi_{\alpha}(x)$ are multivariate orthogonal polynomials (multi-index $\alpha$), and $c_{\alpha}$ are coefficients fit by regression.

Basis selection
- Hermite polynomials are the canonical choice for inputs approximately Gaussian after transforms. For bounded inputs other orthogonal polynomials (Legendre, Jacobi) may be appropriate.

Coefficient estimation
- Ordinary least squares (OLS) or ridge regression for stability. For $M$ basis functions, OLS involves solving $(\Phi^\top\Phi)c=\Phi^\top y$ where $\Phi_{i\alpha}=\Psi_{\alpha}(x_i)$.

Uncertainty quantification
- Under Gaussian noise assumptions and OLS, coefficient covariance is
  $$\mathrm{Cov}(c) = \sigma_n^2 (\Phi^\top\Phi)^{-1}.$$ 
- Predictive variance at $x_*$ is then
  $$\mathrm{Var}(\hat y(x_*)) = \Psi(x_*)^\top \mathrm{Cov}(c) \Psi(x_*).$$

Hyperparameters
- `max_order` — maximum polynomial total order
- `basis` — `hermite` (default) or other families
- `reg` — ridge regularization parameter

Practical notes
- PCE works best when the function is smooth and well-approximated by low-order polynomials; diffractive dips may require higher-order terms or localized bases.
- The number of basis terms grows combinatorially with input dimension and order — careful selection or sparsity-promoting regression (LASSO) may be necessary.

Persistence
- Save coefficients and basis metadata (`basis`, `max_order`, input scaling) using `joblib` or `numpy.save`.

[models.pce]
max_order = 4
basis = "hermite"
reg = 1e-6
```

Example config for quick experiments: [examples/config_pce.toml](../examples/config_pce.toml)
```toml
[models.pce]
max_order = 4
basis = "hermite"
reg = 1e-6
```
[models.pce]
max_order = 4
basis = "hermite"
reg = 1e-6
```

Diagnostics
- Coefficient magnitude plots across orders to detect truncation error
- Predictive variance heatmaps across kinematic slices
