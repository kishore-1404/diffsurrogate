"""Deep Gaussian Process surrogate with Doubly Stochastic Variational Inference.

Two-layer default: each hidden GP layer uses CholeskyVariationalDistribution
+ VariationalStrategy over ``n_inducing`` learnable inducing points with a
Matérn-5/2 ARD kernel, plus a Constant mean. The final layer produces the
scalar output.

Training optimizes the DeepApproximateMLL (ELBO with ``num_samples``
MC draws through the stack) with Adam.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from diffsurrogate.models.base import SurrogateModel

logger = logging.getLogger(__name__)


def _require_gpytorch():
    try:
        import gpytorch  # noqa: F401
        import torch  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "DeepGPSurrogate requires GPyTorch. Install with: "
            "pip install diffsurrogate[gpytorch]"
        ) from e


def _build_dgp(input_dim: int, n_layers: int, n_inducing: int, hidden_dim: int):
    import gpytorch
    from gpytorch.models.deep_gps import DeepGP, DeepGPLayer
    import torch

    class HiddenLayer(DeepGPLayer):
        def __init__(self, input_dims, output_dims, n_induce):
            inducing = torch.randn(output_dims, n_induce, input_dims)
            batch_shape = torch.Size([output_dims])
            vdist = gpytorch.variational.CholeskyVariationalDistribution(
                n_induce, batch_shape=batch_shape
            )
            vstrat = gpytorch.variational.VariationalStrategy(
                self, inducing, vdist, learn_inducing_locations=True
            )
            super().__init__(vstrat, input_dims, output_dims)
            self.mean_module = gpytorch.means.ConstantMean(batch_shape=batch_shape)
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.MaternKernel(
                    nu=2.5, batch_shape=batch_shape, ard_num_dims=input_dims
                ),
                batch_shape=batch_shape,
            )

        def forward(self, x):
            return gpytorch.distributions.MultivariateNormal(
                self.mean_module(x), self.covar_module(x)
            )

    class LastLayer(DeepGPLayer):
        def __init__(self, input_dims, n_induce):
            inducing = torch.randn(n_induce, input_dims)
            vdist = gpytorch.variational.CholeskyVariationalDistribution(n_induce)
            vstrat = gpytorch.variational.VariationalStrategy(
                self, inducing, vdist, learn_inducing_locations=True
            )
            super().__init__(vstrat, input_dims, None)
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=input_dims)
            )

        def forward(self, x):
            return gpytorch.distributions.MultivariateNormal(
                self.mean_module(x), self.covar_module(x)
            )

    class DGP(DeepGP):
        def __init__(self):
            super().__init__()
            self.hidden_layers = torch.nn.ModuleList()
            cur_dim = input_dim
            for _ in range(max(1, n_layers - 1)):
                h = HiddenLayer(cur_dim, hidden_dim, n_inducing)
                self.hidden_layers.append(h)
                cur_dim = hidden_dim
            self.last = LastLayer(cur_dim, n_inducing)
            self.likelihood = gpytorch.likelihoods.GaussianLikelihood()

        def forward(self, x):
            h = x
            for layer in self.hidden_layers:
                h = layer(h)
            return self.last(h)

    return DGP()


class DeepGPSurrogate(SurrogateModel):
    """DeepGP surrogate via GPyTorch / DSVI."""

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.model = None
        self._input_dim: int | None = None

    def name(self) -> str:
        return "deep_gp"

    def supports_uq(self) -> bool:
        return True

    # ------------------------------------------------------------------ fit
    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        _require_gpytorch()
        import gpytorch
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        X_train = self._as_2d(X_train)
        y_train = self._as_1d(y_train)
        self._input_dim = X_train.shape[1]

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        hidden_dim = max(2, self._input_dim)
        self.model = _build_dgp(
            input_dim=self._input_dim,
            n_layers=self.cfg.n_layers,
            n_inducing=self.cfg.n_inducing,
            hidden_dim=hidden_dim,
        ).to(device)

        self.model.train()
        self.model.likelihood.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr)
        mll = gpytorch.mlls.DeepApproximateMLL(
            gpytorch.mlls.VariationalELBO(
                self.model.likelihood, self.model, num_data=X_train.shape[0]
            )
        )

        ds = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float())
        loader = DataLoader(ds, batch_size=min(self.cfg.batch_size, X_train.shape[0]), shuffle=True)

        for epoch in range(self.cfg.max_epochs):
            for xb, yb in loader:
                xb = xb.to(device)
                yb = yb.to(device)
                with gpytorch.settings.num_likelihood_samples(self.cfg.n_samples):
                    optimizer.zero_grad()
                    output = self.model(xb)
                    loss = -mll(output, yb)
                    loss.backward()
                    optimizer.step()
            if epoch % 20 == 0:
                logger.debug("DeepGP epoch %d  loss=%.4e", epoch, float(loss.detach().cpu()))

        self.model.eval()
        self.model.likelihood.eval()

    # -------------------------------------------------------------- predict
    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        if self.model is None:
            raise RuntimeError("DeepGP.predict called before fit/load")
        import gpytorch
        import torch

        X = self._as_2d(X)
        device = next(self.model.parameters()).device
        xt = torch.from_numpy(X).float().to(device)
        self.model.eval()
        self.model.likelihood.eval()
        with torch.no_grad(), gpytorch.settings.num_likelihood_samples(self.cfg.n_samples):
            pred = self.model.likelihood(self.model(xt))
            # pred.mean shape: (S, N); average over MC samples.
            mean = pred.mean.mean(0).cpu().numpy().ravel()
            # Variance = mean of variances + variance of means (total variance).
            within = pred.variance.mean(0)
            between = pred.mean.var(0)
            var = (within + between).cpu().numpy().ravel()
        std = np.sqrt(np.maximum(var, 0.0))
        return mean, std

    # ---------------------------------------------------------------- save
    def save(self, path: Path) -> None:
        if self.model is None:
            raise RuntimeError("DeepGP.save called before fit")
        import torch

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path / "dgp_state.pt")
        meta = {
            "input_dim": self._input_dim,
            "n_layers": self.cfg.n_layers,
            "n_inducing": self.cfg.n_inducing,
            "hidden_dim": max(2, self._input_dim),
        }
        with (path / "dgp_meta.json").open("w") as f:
            json.dump(meta, f)

    # ---------------------------------------------------------------- load
    def load(self, path: Path) -> None:
        _require_gpytorch()
        import torch

        path = Path(path)
        with (path / "dgp_meta.json").open() as f:
            meta = json.load(f)
        self._input_dim = meta["input_dim"]
        self.model = _build_dgp(
            input_dim=meta["input_dim"],
            n_layers=meta["n_layers"],
            n_inducing=meta["n_inducing"],
            hidden_dim=meta["hidden_dim"],
        )
        self.model.load_state_dict(torch.load(path / "dgp_state.pt", map_location="cpu"))
        self.model.eval()
        self.model.likelihood.eval()
