"""Standard NN surrogate: MLP / ResNet with early stopping.

Matches the paper's baseline ``3-30-30-30-1`` topology by default, with
``tanh`` activation and residual skip connections. Optimizer: Adam with
weight decay. Early stopping monitors validation loss on a 10% internal
holdout that's carved out of the training split (separate from any
benchmark test split).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from diffsurrogate.models.base import SurrogateModel

logger = logging.getLogger(__name__)


def _require_torch():
    try:
        import torch  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "PyTorch is required for NeuralNetSurrogate. "
            "Install with: pip install diffsurrogate[torch]"
        ) from e


def _activation_module(name: str):
    import torch.nn as nn

    name = name.lower()
    return {
        "tanh": nn.Tanh(),
        "relu": nn.ReLU(),
        "gelu": nn.GELU(),
        "swish": nn.SiLU(),
        "silu": nn.SiLU(),
    }[name]


class _MLPCore:
    """Core MLP / ResNet module factory (built only when torch is available)."""

    @staticmethod
    def build(input_dim: int, hidden_layers: list[int], activation: str, use_resnet: bool):
        import torch.nn as nn

        class SartreNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.use_resnet = use_resnet
                self.entry = nn.Sequential(
                    nn.Linear(input_dim, hidden_layers[0]),
                    _activation_module(activation),
                )
                self.blocks = nn.ModuleList()
                for i in range(1, len(hidden_layers)):
                    in_d = hidden_layers[i - 1]
                    out_d = hidden_layers[i]
                    block = nn.Sequential(nn.Linear(in_d, out_d), _activation_module(activation))
                    self.blocks.append(block)
                self.head = nn.Linear(hidden_layers[-1], 1)

            def forward(self, x):
                h = self.entry(x)
                for blk in self.blocks:
                    # Residual only when shapes match.
                    out = blk(h)
                    if self.use_resnet and out.shape == h.shape:
                        h = h + out
                    else:
                        h = out
                return self.head(h)

        return SartreNet()


class NeuralNetSurrogate(SurrogateModel):
    """MLP / ResNet surrogate with early stopping."""

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.model = None
        self._input_dim: int | None = None

    def name(self) -> str:
        return "neural_net"

    def supports_uq(self) -> bool:
        return False

    # ------------------------------------------------------------------ fit
    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        _require_torch()
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        X_train = self._as_2d(X_train)
        y_train = self._as_1d(y_train).reshape(-1, 1)
        self._input_dim = X_train.shape[1]

        # Internal 10% holdout from training data for early stopping.
        rng = np.random.default_rng(42)
        n = X_train.shape[0]
        perm = rng.permutation(n)
        n_val = max(1, int(0.1 * n))
        val_idx = perm[:n_val]
        tr_idx = perm[n_val:]
        Xt, yt = X_train[tr_idx], y_train[tr_idx]
        Xv, yv = X_train[val_idx], y_train[val_idx]

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = _MLPCore.build(
            input_dim=self._input_dim,
            hidden_layers=self.cfg.hidden_layers,
            activation=self.cfg.activation,
            use_resnet=self.cfg.use_resnet,
        ).to(device)
        opt = torch.optim.Adam(
            self.model.parameters(),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
        )
        loss_fn = torch.nn.MSELoss()

        ds = TensorDataset(torch.from_numpy(Xt).float(), torch.from_numpy(yt).float())
        loader = DataLoader(ds, batch_size=self.cfg.batch_size, shuffle=True, drop_last=False)
        Xv_t = torch.from_numpy(Xv).float().to(device)
        yv_t = torch.from_numpy(yv).float().to(device)

        best_val = float("inf")
        patience_ctr = 0
        best_state = None
        for epoch in range(self.cfg.max_epochs):
            self.model.train()
            for xb, yb in loader:
                xb = xb.to(device)
                yb = yb.to(device)
                opt.zero_grad()
                pred = self.model(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                opt.step()

            self.model.eval()
            with torch.no_grad():
                val_pred = self.model(Xv_t)
                val_loss = loss_fn(val_pred, yv_t).item()

            if val_loss < best_val - 1e-9:
                best_val = val_loss
                patience_ctr = 0
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                patience_ctr += 1
                if patience_ctr >= self.cfg.early_stopping_patience:
                    logger.info("NN early stopping at epoch %d (val=%.4e)", epoch, best_val)
                    break

            if epoch % 50 == 0:
                logger.debug("NN epoch %d  val_loss=%.4e  best=%.4e", epoch, val_loss, best_val)

        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.model.eval()

    # -------------------------------------------------------------- predict
    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        if self.model is None:
            raise RuntimeError("NeuralNetSurrogate.predict called before fit/load")
        import torch

        X = self._as_2d(X)
        device = next(self.model.parameters()).device
        self.model.eval()
        with torch.no_grad():
            out = self.model(torch.from_numpy(X).float().to(device)).cpu().numpy().ravel()
        return out, None

    # --------------------------------------------------------------- save
    def save(self, path: Path) -> None:
        if self.model is None:
            raise RuntimeError("NeuralNetSurrogate.save called before fit")
        import torch

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path / "weights.pt")
        meta = {
            "input_dim": self._input_dim,
            "hidden_layers": self.cfg.hidden_layers,
            "activation": self.cfg.activation,
            "use_resnet": self.cfg.use_resnet,
        }
        with (path / "nn_meta.json").open("w") as f:
            json.dump(meta, f, indent=2)

    # --------------------------------------------------------------- load
    def load(self, path: Path) -> None:
        _require_torch()
        import torch

        path = Path(path)
        with (path / "nn_meta.json").open() as f:
            meta = json.load(f)
        self._input_dim = meta["input_dim"]
        self.model = _MLPCore.build(
            input_dim=meta["input_dim"],
            hidden_layers=meta["hidden_layers"],
            activation=meta["activation"],
            use_resnet=meta["use_resnet"],
        )
        state = torch.load(path / "weights.pt", map_location="cpu")
        self.model.load_state_dict(state)
        self.model.eval()
