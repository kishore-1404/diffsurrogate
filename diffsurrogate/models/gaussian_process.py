"""Gaussian process surrogate.

Default: exact GP with Matérn-5/2 ARD kernel and a WhiteNoise term, fit via
scikit-learn's ``GaussianProcessRegressor`` (L-BFGS-B with multi-restart).

Auto-switch: if ``N_train > sparse_threshold`` (or ``sparse=true``), we switch
to a sparse variational GP (SVGP) implemented via GPyTorch. GPyTorch is an
optional dependency; if it isn't installed, we fall back to scikit-learn with
a warning even at large N.
"""

from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path

import joblib
import numpy as np

from diffsurrogate.models.base import SurrogateModel

logger = logging.getLogger(__name__)


def _build_sklearn_kernel(kind: str, use_white_noise: bool, input_dim: int):
    """Build a sklearn ARD kernel with sensible bounds for scaled inputs.

    Because our inputs are z-scored to roughly [-3, 3], realistic length
    scales live in ~[1e-2, 1e2]. We tighten the default sklearn bounds
    (1e-5, 1e5) so the L-BFGS-B optimizer doesn't clamp to the boundary
    — this was causing the t-axis length scale to collapse to 1e-5 and
    the model to output a constant. Similarly we bound the signal variance
    and noise level to physically reasonable ranges for unit-variance data.
    """
    from sklearn.gaussian_process.kernels import (
        RBF,
        ConstantKernel,
        Matern,
        WhiteKernel,
    )

    length_scale = np.ones(input_dim)
    ls_bounds = (1e-2, 1e2)
    sig_bounds = (1e-3, 1e3)
    if kind == "matern52_ard":
        base = ConstantKernel(1.0, sig_bounds) * Matern(
            length_scale=length_scale, length_scale_bounds=ls_bounds, nu=2.5
        )
    elif kind == "matern32_ard":
        base = ConstantKernel(1.0, sig_bounds) * Matern(
            length_scale=length_scale, length_scale_bounds=ls_bounds, nu=1.5
        )
    elif kind == "rbf_ard":
        base = ConstantKernel(1.0, sig_bounds) * RBF(
            length_scale=length_scale, length_scale_bounds=ls_bounds
        )
    else:
        raise ValueError(f"Unknown kernel '{kind}'")
    if use_white_noise:
        base = base + WhiteKernel(noise_level=1e-4, noise_level_bounds=(1e-8, 1e0))
    return base


class GaussianProcessSurrogate(SurrogateModel):
    """Exact or sparse variational GP, auto-selected by dataset size."""

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self._sklearn_model = None  # sklearn GPR (exact mode)
        self._svgp = None  # GPyTorch SVGP model (sparse mode)
        self._svgp_likelihood = None
        self._svgp_meta: dict | None = None
        self._mode: str = "exact"

    def name(self) -> str:
        return "gaussian_process"

    def supports_uq(self) -> bool:
        return True

    # ------------------------------------------------------------------ fit
    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        X_train = self._as_2d(X_train)
        y_train = self._as_1d(y_train)
        n = X_train.shape[0]

        use_sparse = self.cfg.sparse or n > self.cfg.sparse_threshold
        if use_sparse:
            try:
                import torch  # noqa: F401
                import gpytorch  # noqa: F401

                self._mode = "svgp"
                self._fit_svgp(X_train, y_train)
                return
            except ImportError:
                logger.warning(
                    "N=%d exceeds sparse_threshold=%d but GPyTorch is not installed; "
                    "falling back to exact scikit-learn GP (will be slow).",
                    n,
                    self.cfg.sparse_threshold,
                )
        self._mode = "exact"
        self._fit_exact(X_train, y_train)

    def _fit_exact(self, X: np.ndarray, y: np.ndarray) -> None:
        from sklearn.gaussian_process import GaussianProcessRegressor

        kernel = _build_sklearn_kernel(self.cfg.kernel, self.cfg.use_white_noise, X.shape[1])
        model = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=self.cfg.n_restarts,
            normalize_y=False,
            alpha=1e-10,
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=Warning)
            model.fit(X, y)
        self._sklearn_model = model
        logger.info("Exact GP fitted; learned kernel: %s", model.kernel_)

    def _fit_svgp(self, X: np.ndarray, y: np.ndarray) -> None:
        import gpytorch
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n, d = X.shape
        n_induce = min(self.cfg.n_inducing, n)
        idx = np.random.default_rng(0).choice(n, size=n_induce, replace=False)
        inducing_points = torch.from_numpy(X[idx]).float().to(device)

        # Pick kernel family for GPyTorch.
        if self.cfg.kernel == "matern52_ard":
            base_kernel = gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=d)
        elif self.cfg.kernel == "matern32_ard":
            base_kernel = gpytorch.kernels.MaternKernel(nu=1.5, ard_num_dims=d)
        else:
            base_kernel = gpytorch.kernels.RBFKernel(ard_num_dims=d)

        class SVGPModel(gpytorch.models.ApproximateGP):
            def __init__(self, inducing):
                vdist = gpytorch.variational.CholeskyVariationalDistribution(inducing.size(0))
                vstrat = gpytorch.variational.VariationalStrategy(
                    self, inducing, vdist, learn_inducing_locations=True
                )
                super().__init__(vstrat)
                self.mean_module = gpytorch.means.ConstantMean()
                self.covar_module = gpytorch.kernels.ScaleKernel(base_kernel)

            def forward(self, x):
                return gpytorch.distributions.MultivariateNormal(
                    self.mean_module(x), self.covar_module(x)
                )

        model = SVGPModel(inducing_points).to(device)
        likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
        model.train()
        likelihood.train()

        optimizer = torch.optim.Adam(
            [{"params": model.parameters()}, {"params": likelihood.parameters()}],
            lr=0.01,
        )
        mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=n)

        ds = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).float())
        loader = DataLoader(ds, batch_size=min(1024, n), shuffle=True)
        max_epochs = max(50, 100)
        for epoch in range(max_epochs):
            for xb, yb in loader:
                xb = xb.to(device)
                yb = yb.to(device)
                optimizer.zero_grad()
                output = model(xb)
                loss = -mll(output, yb)
                loss.backward()
                optimizer.step()
            if epoch % 20 == 0:
                logger.debug("SVGP epoch %d  loss=%.4e", epoch, float(loss.detach().cpu()))

        model.eval()
        likelihood.eval()
        self._svgp = model
        self._svgp_likelihood = likelihood
        self._svgp_meta = {"n_inducing": int(n_induce), "input_dim": int(d), "kernel": self.cfg.kernel}
        logger.info("SVGP fitted with %d inducing points", n_induce)

    # -------------------------------------------------------------- predict
    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        X = self._as_2d(X)
        if self._mode == "exact":
            if self._sklearn_model is None:
                raise RuntimeError("GP.predict called before fit/load")
            mean, std = self._sklearn_model.predict(X, return_std=True)
            return mean.ravel(), std.ravel()
        # SVGP
        import gpytorch
        import torch

        assert self._svgp is not None and self._svgp_likelihood is not None
        device = next(self._svgp.parameters()).device
        xt = torch.from_numpy(X).float().to(device)
        self._svgp.eval()
        self._svgp_likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred = self._svgp_likelihood(self._svgp(xt))
            mean = pred.mean.cpu().numpy().ravel()
            std = pred.stddev.cpu().numpy().ravel()
        return mean, std

    # --------------------------------------------------------------- save
    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        with (path / "gp_mode.json").open("w") as f:
            json.dump({"mode": self._mode}, f)
        if self._mode == "exact":
            if self._sklearn_model is None:
                raise RuntimeError("Nothing to save; fit first")
            joblib.dump(self._sklearn_model, path / "sklearn_gp.joblib")
        else:
            import torch

            if self._svgp is None:
                raise RuntimeError("Nothing to save; fit first")
            torch.save(self._svgp.state_dict(), path / "svgp_state.pt")
            torch.save(self._svgp_likelihood.state_dict(), path / "svgp_likelihood.pt")
            with (path / "svgp_meta.json").open("w") as f:
                json.dump(self._svgp_meta or {}, f)
            # Save inducing points initial location (state_dict has them but
            # we need shape info to reconstruct the module).

    # --------------------------------------------------------------- load
    def load(self, path: Path) -> None:
        path = Path(path)
        with (path / "gp_mode.json").open() as f:
            self._mode = json.load(f)["mode"]
        if self._mode == "exact":
            self._sklearn_model = joblib.load(path / "sklearn_gp.joblib")
        else:
            import gpytorch
            import torch

            with (path / "svgp_meta.json").open() as f:
                meta = json.load(f)
            self._svgp_meta = meta
            d = meta["input_dim"]
            n_induce = meta["n_inducing"]
            inducing_points = torch.zeros(n_induce, d)

            if meta["kernel"] == "matern52_ard":
                base_kernel = gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=d)
            elif meta["kernel"] == "matern32_ard":
                base_kernel = gpytorch.kernels.MaternKernel(nu=1.5, ard_num_dims=d)
            else:
                base_kernel = gpytorch.kernels.RBFKernel(ard_num_dims=d)

            class SVGPModel(gpytorch.models.ApproximateGP):
                def __init__(self, inducing):
                    vdist = gpytorch.variational.CholeskyVariationalDistribution(inducing.size(0))
                    vstrat = gpytorch.variational.VariationalStrategy(
                        self, inducing, vdist, learn_inducing_locations=True
                    )
                    super().__init__(vstrat)
                    self.mean_module = gpytorch.means.ConstantMean()
                    self.covar_module = gpytorch.kernels.ScaleKernel(base_kernel)

                def forward(self, x):
                    return gpytorch.distributions.MultivariateNormal(
                        self.mean_module(x), self.covar_module(x)
                    )

            self._svgp = SVGPModel(inducing_points)
            self._svgp.load_state_dict(torch.load(path / "svgp_state.pt", map_location="cpu"))
            self._svgp_likelihood = gpytorch.likelihoods.GaussianLikelihood()
            self._svgp_likelihood.load_state_dict(
                torch.load(path / "svgp_likelihood.pt", map_location="cpu")
            )
            self._svgp.eval()
            self._svgp_likelihood.eval()
