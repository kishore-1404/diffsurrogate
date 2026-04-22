# Deep Gaussian Process (Deep GP)

Deep Gaussian Processes (DGPs) are hierarchical compositions of GPs that allow rich non-Gaussian, non-stationary function priors. This page covers architecture, inference, and practical usage in `diffsurrogate`.

Model structure
- A two-layer DGP can be written as:
  $$h(x) \sim \mathcal{GP}_1(0, k_1), \qquad f(h) \sim \mathcal{GP}_2(0,k_2),$$
  and observations $y=f(h(x))+\epsilon,\ \epsilon\sim\mathcal{N}(0,\sigma_n^2)$.

Inference strategy
- Due to the intractability of the exact posterior, `diffsurrogate` uses doubly-stochastic variational inference (DSVI). The method introduces variational distributions for the layer outputs/inducing points and uses Monte Carlo for unbiased gradient estimates.

Training
- The objective is the ELBO; optimization is performed with Adam and minibatching. Typical training requires more epochs than single-layer GPs because of additional variational parameters.

Hyperparameters
- `n_layers` (typical: 2)
- `n_inducing_per_layer` (e.g., 64–256)
- `n_samples_predict` — MC samples to average at prediction time (e.g., 100)
- `lr`, `batch_size`, `epochs`

Prediction
- Predictive mean is estimated by sampling $S$ forward passes through variational layers and averaging:
  $$\hat y = \frac{1}{S} \sum_{s=1}^S f^{(s)}(x), \quad s_{\hat y}^2 = \frac{1}{S-1}\sum (f^{(s)}-\hat y)^2.$$ 

Persistence
- Save model state and `metadata.json` with `n_samples_predict` and variational hyperparameters used during training.

Notes
- Expect some stochasticity across runs even with fixed seeds due to MC approximations; increase `n_samples_predict` for stable estimates at inference cost.

Pitfalls
- Poorly initialized inducing points slow convergence: initialize with KMeans on inputs of each layer.

Example config
```toml
[models.deep_gp]
n_layers = 2
n_inducing_per_layer = 128
n_samples_predict = 200
lr = 1e-3
epochs = 800
```

Example config for quick experiments: [examples/config_deep_gp.toml](../examples/config_deep_gp.toml)
