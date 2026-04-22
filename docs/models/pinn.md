# Physics-Informed Neural Network (PINN)

This page explains the `pinn` surrogate: how physics constraints are integrated into training, common equations used, and best practices.

Purpose
- PINNs embed known physical constraints (e.g., evolution equations, conservation laws, unitarity) into the loss function so that the learned surrogate respects domain knowledge beyond data points.

Loss composition
- The typical composite loss is:
  $$\mathcal{L} = \lambda_{\text{data}} \frac{1}{N_d}\sum_{i=1}^{N_d} (y_i - \hat y_i)^2 + \lambda_{\text{phys}} \frac{1}{N_p}\sum_{j=1}^{N_p} R(x_j)^2 + \lambda_{\text{reg}} \|\theta\|^2$$
  where $R(x)$ is the physics residual evaluated at collocation points.

Example physics residuals
- Unitarity penalty: if $A$ must satisfy $|A|\leq 1$ (example), penalize excess magnitude:
  $$R_{\text{unit}}(x) = \max\bigl(0, |\widehat{A}(x)| - 1\bigr).$$
- BK evolution (schematic): if $\mathcal{B}[A]=0$ is the evolution PDE, residual is $R(x)=\mathcal{B}[\widehat{A}(x)]$ evaluated via automatic differentiation in the network.

Collocation & sampling
- Choose collocation points $\{x_j\}$ densely in regions where physics constraints are critical (e.g., near diffractive dips).

Hyperparameters & tuning
- `lambda_phys` — balance between data fit and physics; start small (0.1) and increase until physics residuals are acceptable without destroying data fit.
- `n_collocation` — number of collocation points; larger increases training cost.

Training tips
- Normalize the physics residual to comparable scale as data loss (e.g., divide residuals by characteristic magnitude).
- Use multi-stage training: pretrain on data only, then fine-tune with physics loss added gradually.

UQ
- PINNs do not inherently provide UQ. Combine with ensemble or Bayesian NN techniques for uncertainty estimates.

Persistence
- Save model state and `metadata.json` containing `lambda_phys`, `n_collocation`, and collocation sampling strategy.

Pitfalls
- Weighting mismatch: if `lambda_phys` is too large, model may ignore data; too small and physics is ineffective. Monitor both data loss and physics residual separately during training.

[models.pinn]
lambda_phys = 0.1
lambda_reg = 1e-6
n_collocation = 5000
pretrain_epochs = 50
finetune_epochs = 150
```

Example config for quick experiments: [examples/config_pinn.toml](../examples/config_pinn.toml)
```toml
[models.pinn]
lambda_phys = 0.1
lambda_reg = 1e-6
n_collocation = 5000
pretrain_epochs = 50
finetune_epochs = 150
```
[models.pinn]
lambda_phys = 0.1
lambda_reg = 1e-6
n_collocation = 5000
pretrain_epochs = 50
finetune_epochs = 150
```
