"""Physics-Informed Neural Network surrogate.

The target is ln(A) (the log-squared amplitude). We also attach an auxiliary
head that predicts the dipole amplitude N in [0, 1], and impose two physics
losses on this head:

  L_data   = MSE(ln(A)_pred, ln(A)_true)
  L_BK     = MSE residual of the BK evolution equation
             dN / d ln(1/x) ≈ alpha_s * (N - N^2)
  L_bound  = sum[ max(0, N-1)^2 + max(0, -N)^2 ]  (unitarity 0 <= N <= 1)

The BK residual uses a simplified local kernel: we replace the non-local
convolution K ⊗ (N - N²) by its integrand at the collocation point, which is
a standard first-approximation when you want to enforce the *shape* of BK
dynamics without solving the full integro-differential problem inside the
training loop. The paper mandates the structural form of the loss; the
convolution kernel K is left schematic in the code-level pseudocode there
and we implement this standard simplification.

W² is used to compute ln(1/x) ≈ ln(W²/Q²) as the BK evolution variable.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from diffsurrogate.models.base import SurrogateModel
from diffsurrogate.models.neural_net import _activation_module, _require_torch

logger = logging.getLogger(__name__)


class PINNSurrogate(SurrogateModel):
    """PINN with BK residual and unitarity penalties."""

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.model = None
        self._input_dim: int | None = None
        # Scaler info is *not* stored on the model itself — we only need to
        # know that input column 1 (W2) is log-then-z-scored, and column 0
        # (Q2) might or might not be. The PINN works in scaled space, so the
        # BK residual is written in scaled coordinates and the weight lambda
        # absorbs any residual jacobian.

    def name(self) -> str:
        return "pinn"

    def supports_uq(self) -> bool:
        return False

    # --- network definition ---
    def _build_net(self, input_dim: int):
        _require_torch()
        import torch.nn as nn

        hidden = self.cfg.hidden_layers
        activation = self.cfg.activation

        class PINNNet(nn.Module):
            def __init__(self):
                super().__init__()
                layers: list = [nn.Linear(input_dim, hidden[0]), _activation_module(activation)]
                for i in range(1, len(hidden)):
                    layers += [nn.Linear(hidden[i - 1], hidden[i]), _activation_module(activation)]
                self.body = nn.Sequential(*layers)
                self.lnA_head = nn.Linear(hidden[-1], 1)
                # N-head bounded to (0, 1) via sigmoid so unitarity is cheap.
                self.N_head = nn.Sequential(nn.Linear(hidden[-1], 1), nn.Sigmoid())

            def forward(self, x):
                h = self.body(x)
                return self.lnA_head(h), self.N_head(h)

        return PINNNet()

    # ------------------------------------------------------------------ fit
    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        _require_torch()
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        X_train = self._as_2d(X_train)
        y_train = self._as_1d(y_train).reshape(-1, 1)
        self._input_dim = X_train.shape[1]

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._build_net(self._input_dim).to(device)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr)

        # Collocation points: uniform inside the training hypercube.
        X_min = X_train.min(axis=0)
        X_max = X_train.max(axis=0)
        rng = np.random.default_rng(123)
        n_coll = self.cfg.n_collocation
        coll = rng.uniform(X_min, X_max, size=(n_coll, self._input_dim))

        ds = TensorDataset(
            torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float()
        )
        loader = DataLoader(ds, batch_size=self.cfg.batch_size, shuffle=True, drop_last=False)
        coll_t = torch.from_numpy(coll).float().to(device).requires_grad_(True)

        alpha_s = float(self.cfg.alpha_s)
        lam_pde = float(self.cfg.lambda_pde)
        lam_bnd = float(self.cfg.lambda_boundary)

        for epoch in range(self.cfg.max_epochs):
            self.model.train()
            for xb, yb in loader:
                xb = xb.to(device)
                yb = yb.to(device)
                opt.zero_grad()

                # Data loss on ln(A) head.
                lnA_pred, _ = self.model(xb)
                loss_data = torch.mean((lnA_pred - yb) ** 2)

                # Physics loss on N head at collocation points.
                _, N_coll = self.model(coll_t)
                # dN / d(ln 1/x): we use column 1 (W2) as the evolution
                # variable in scaled space since ln(1/x) is monotone in ln W2.
                grad_outputs = torch.ones_like(N_coll)
                grads = torch.autograd.grad(
                    N_coll, coll_t, grad_outputs=grad_outputs,
                    create_graph=True, retain_graph=True,
                )[0]
                dN_dy = grads[:, 1:2]
                bk_residual = dN_dy - alpha_s * (N_coll - N_coll ** 2)
                loss_pde = torch.mean(bk_residual ** 2)

                # Unitarity: N-head is sigmoid-bounded so these are
                # typically near zero, but we keep the penalty for safety
                # when the sigmoid saturates.
                loss_bnd = torch.mean(
                    torch.relu(N_coll - 1.0) ** 2 + torch.relu(-N_coll) ** 2
                )

                loss = loss_data + lam_pde * loss_pde + lam_bnd * loss_bnd
                loss.backward()
                opt.step()

            if epoch % 100 == 0:
                logger.debug(
                    "PINN epoch %d  data=%.4e pde=%.4e bnd=%.4e",
                    epoch,
                    float(loss_data.detach().cpu()),
                    float(loss_pde.detach().cpu()),
                    float(loss_bnd.detach().cpu()),
                )

        self.model.eval()

    # -------------------------------------------------------------- predict
    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        if self.model is None:
            raise RuntimeError("PINNSurrogate.predict called before fit/load")
        import torch

        X = self._as_2d(X)
        device = next(self.model.parameters()).device
        self.model.eval()
        with torch.no_grad():
            lnA_pred, _ = self.model(torch.from_numpy(X).float().to(device))
        return lnA_pred.cpu().numpy().ravel(), None

    # ---------------------------------------------------------------- save
    def save(self, path: Path) -> None:
        if self.model is None:
            raise RuntimeError("PINN.save called before fit")
        import torch

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path / "weights.pt")
        meta = {
            "input_dim": self._input_dim,
            "hidden_layers": self.cfg.hidden_layers,
            "activation": self.cfg.activation,
        }
        with (path / "pinn_meta.json").open("w") as f:
            json.dump(meta, f, indent=2)

    # ---------------------------------------------------------------- load
    def load(self, path: Path) -> None:
        _require_torch()
        import torch

        path = Path(path)
        with (path / "pinn_meta.json").open() as f:
            meta = json.load(f)
        self._input_dim = meta["input_dim"]
        # Rebuild with whatever hidden_layers were saved (may differ from cfg).
        orig_hidden = self.cfg.hidden_layers
        orig_act = self.cfg.activation
        self.cfg.hidden_layers = meta["hidden_layers"]
        self.cfg.activation = meta["activation"]
        try:
            self.model = self._build_net(self._input_dim)
        finally:
            self.cfg.hidden_layers = orig_hidden
            self.cfg.activation = orig_act
        state = torch.load(path / "weights.pt", map_location="cpu")
        self.model.load_state_dict(state)
        self.model.eval()
