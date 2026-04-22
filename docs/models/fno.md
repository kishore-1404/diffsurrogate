# Fourier Neural Operator (FNO-1d over t)

FNO learns operators mapping functions to functions using Fourier-domain multipliers. In `diffsurrogate` we apply FNO along the $t$-axis when the dataset is gridded in $t$ for each $(Q^2,W^2)$ slice.

When to use
- Use FNO when the lookup table provides samples $u_j = y(Q^2,W^2,t_j)$ on a common ordered grid $t_1\dots t_L$ across slices. FNO excels at learning global patterns along the grid.

Mathematical sketch
- For a single slice, data is a vector $u\in\mathbb{R}^L$. FNO layers act by:
  $$u_{\ell+1} = \sigma\Bigl( \mathcal{F}^{-1} \bigl( P_{\ell}(k) \cdot \mathcal{F}(u_{\ell}) (k) \bigr) + W_{\ell} u_{\ell} \Bigr)$$
  where $\mathcal{F}$ is the DFT across the $t$ grid, $P_{\ell}(k)$ are learned spectral multipliers, and $W_{\ell}$ is a local linear mapping.

Architectural tips
- Choose `modes` (number of low-frequency modes to keep) based on grid resolution and expected spectral content.
- Use padding or windowing if the grid has endpoints with special behaviour.

Detection of gridded data
- Loader checks that for each unique $(Q^2,W^2)$ pair, the sorted `t_center` arrays are identical. If not, FNO branch is disabled and fallback MLP is used.

Hyperparameters
- `modes`: number of Fourier modes retained (e.g., 8–32)
- `width`: channel width (e.g., 64)
- `n_layers`: number of spectral layers
- `lr`, `batch_size`, `epochs`

Training notes
- Use spectral normalization and weight decay to stabilize training when high-frequency modes are learned.
- Monitor per-slice reconstruction error to ensure the operator generalizes across different $(Q^2,W^2)$ contexts.

Persistence
- Save FNO weights as PyTorch state dict and `metadata.json` with grid specification (t values and ordering) so inference reconstructs per-slice inputs in the same shape.

Pitfalls
- Non-uniform or missing t-grid values across slices will break the operator assumption. Ensure data is preprocessed to a common grid or use interpolation to resample.
- Overfitting high-frequency modes can create oscillatory artefacts—prefer small `modes` and regularization.

[models.fno]
modes = 16
width = 64
n_layers = 4
lr = 5e-4
epochs = 300
```

Example config for quick experiments: [examples/config_fno.toml](../examples/config_fno.toml)
```toml
[models.fno]
modes = 16
width = 64
n_layers = 4
lr = 5e-4
epochs = 300
```
[models.fno]
modes = 16
width = 64
n_layers = 4
lr = 5e-4
epochs = 300
```
