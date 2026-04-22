# Neural Network (ResNet-style) surrogate

This page describes the `neural_net` surrogate in detail: architecture, math, training recipe, hyperparameters, persistence format, and practical tips.

Overview
- Deterministic feedforward residual network used as a fast baseline surrogate for $y=\ln A(Q^2,W^2,t)$.
- Good for large datasets and low-latency inference.

Mathematical model
- Let $x\in\mathbb{R}^D$ be the scaled input vector (features). The neural net defines a parametric mapping $f_{\theta}:\mathbb{R}^D\to\mathbb{R}$ with layer-wise residual blocks:
  $$h^{(0)} = x,\qquad h^{(\ell+1)} = h^{(\ell)} + \mathcal{B}_\ell(h^{(\ell)}),$$
  where $\mathcal{B}_\ell$ is a small feedforward block: linear → activation → linear → optional dropout.
- Output: $\hat y = f_{\theta}(x) = w_{\text{out}}^T h^{(L)} + b_{\text{out}}$.

Loss and optimization
- Default: Mean-squared error (MSE) on scaled targets
  $$\mathcal{L}(\theta) = \frac{1}{N} \sum_{i=1}^N (y_i - f_{\theta}(x_i))^2 + \lambda_{\text{reg}} \|\theta\|^2.$$ 
- Optimizers: Adam (default) or SGD with momentum. Learning-rate scheduling recommended (ReduceLROnPlateau or cosine annealing).

Hyperparameters (recommended defaults)
- `hidden_sizes`: [128,128,64]
- `n_blocks`: 3
- `activation`: `elu`
- `dropout`: 0.0 (0.1 for small datasets)
- `lr`: 1e-3
- `batch_size`: 64
- `epochs`: 200
- `early_stopping_patience`: 20
- `weight_decay` (`lambda_reg`): 1e-6

Regularization & generalization
- Use `robust` target scaling to reduce sensitivity to diffractive dip outliers.
- Data augmentation: small additive noise on input features (within physical tolerance) can improve robustness.

Uncertainty estimation options
- MC Dropout: enable dropout at inference, run $M$ stochastic forward passes, compute sample mean and variance:
  $$\hat y = \frac{1}{M} \sum_{m=1}^M f_{\theta}^{(m)}(x),\quad s^2 = \frac{1}{M-1} \sum_m (f^{(m)}-\hat y)^2.$$ 
- Ensembles: train $K$ networks with different seeds/hyperparams and combine predictions.

Persistence
- Save model `state_dict()` (PyTorch) to `saved_models/neural_net/production/weights.pt` and save `scalers.joblib` alongside `metadata.json` documenting architecture and training config.

Diagnostics and plots
- Training/validation loss curves
- Residual vs $|t|$ plots to see where dips are missed
- Error histograms and quantile-quantile (QQ) plots for residuals

Pitfalls and troubleshooting
- If network outputs are nearly constant: check target scaling and optimizer (too-large weight decay or tiny lr).
- If validation loss diverges: reduce lr, check for data leakage, or scale inputs.

Example config snippet
```toml
[models.neural_net]
hidden_sizes = [128,128,64]
n_blocks = 3
activation = "elu"
dropout = 0.1
lr = 1e-3
batch_size = 64
epochs = 200
early_stopping_patience = 20
weight_decay = 1e-6
```

See also: [docs/models.md](models.md) for conceptual context and performance tips.

Example config for quick experiments: [examples/config_neural_net.toml](../examples/config_neural_net.toml)
