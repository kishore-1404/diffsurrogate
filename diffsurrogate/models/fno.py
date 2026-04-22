"""Fourier Neural Operator surrogate.

The paper defines FNO over spatial density fields; our data is tabular
(Q2, W2, t) scalar triples. Primary strategy:

  1. Detect whether the dataset is gridded on t: i.e., for each unique
     (Q2_key, W2_key), the t values form a reasonably-shared grid.
  2. If so, train an FNO-1d that maps a function defined over t (given
     (Q2, W2) as a constant channel) to lnA over the same t-grid.
  3. If the data isn't gridded, emit a warning and fall back to the same
     ResNet-style MLP as neural_net (marked as "fno_fallback" in the
     metadata so you can see it in the results).

Optional dependency: we only need PyTorch; we implement a minimal
``SpectralConv1d`` inline and don't require the external ``neuraloperator``
package. If you do have ``neuraloperator`` installed and want to use its
more featureful implementation, that's a one-line swap inside
``_build_fno1d``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from diffsurrogate.models.base import SurrogateModel
from diffsurrogate.models.neural_net import _MLPCore, _require_torch

logger = logging.getLogger(__name__)


def _detect_grid(X: np.ndarray, tol: float = 1e-6) -> tuple[bool, dict]:
    """Detect a shared t-grid across (Q2, W2) slices.

    Returns (is_gridded, info). ``info`` contains ``t_grid`` (sorted unique t
    values) and ``slice_index`` mapping (q2, w2) pairs to their row indices
    in sorted t-order.
    """
    # Round Q2, W2 to discover unique slices; use a reasonable precision.
    q = np.round(X[:, 0], decimals=8)
    w = np.round(X[:, 1], decimals=8)
    keys = np.stack([q, w], axis=1)
    # Unique combos
    _, inv = np.unique(keys, axis=0, return_inverse=True)

    # Collect t-values per slice.
    t_sets = {}
    for i, k in enumerate(inv):
        t_sets.setdefault(int(k), []).append(X[i, 2])

    # Build reference grid from the most common slice length.
    lengths = [len(v) for v in t_sets.values()]
    if not lengths:
        return False, {}
    # Pick the modal length; require at least half the slices share it.
    most_common_len = max(set(lengths), key=lengths.count)
    if lengths.count(most_common_len) / len(lengths) < 0.5:
        return False, {}

    # Take a reference slice at the modal length.
    ref_slice = sorted(
        (sorted(v) for v in t_sets.values() if len(v) == most_common_len),
        key=lambda s: s[0],
    )[0]
    ref_grid = np.array(ref_slice)

    # Check all modal-length slices agree with the reference within tol.
    slice_index: dict = {}
    for k, ts in t_sets.items():
        if len(ts) != most_common_len:
            return False, {}
        ts_sorted = np.sort(np.asarray(ts))
        if not np.allclose(ts_sorted, ref_grid, atol=tol, rtol=tol):
            return False, {}
        slice_index[k] = None  # just record presence; caller recomputes idxs

    return True, {"t_grid": ref_grid, "n_slices": len(t_sets)}


class _SpectralConv1d:
    """Factory for a complex-weighted FFT-domain linear layer."""

    @staticmethod
    def build(in_c: int, out_c: int, modes: int):
        import torch
        import torch.nn as nn

        class SpectralConv1d(nn.Module):
            def __init__(self):
                super().__init__()
                self.in_c = in_c
                self.out_c = out_c
                self.modes = modes
                scale = 1.0 / (in_c * out_c)
                self.weights = nn.Parameter(
                    scale * torch.randn(in_c, out_c, modes, 2)
                )  # real & imag

            def forward(self, x):
                # x: (B, C, L)
                B, C, L = x.shape
                x_ft = torch.fft.rfft(x, dim=-1)
                out_ft = torch.zeros(
                    B, self.out_c, x_ft.size(-1),
                    dtype=torch.cfloat, device=x.device,
                )
                m = min(self.modes, x_ft.size(-1))
                w = torch.view_as_complex(self.weights[:, :, :m])
                out_ft[:, :, :m] = torch.einsum("bil,iol->bol", x_ft[:, :, :m], w)
                return torch.fft.irfft(out_ft, n=L, dim=-1)

        return SpectralConv1d()


class _FNO1d:
    @staticmethod
    def build(n_channels_in: int, width: int, n_modes: int, n_layers: int):
        import torch
        import torch.nn as nn

        class FNO1d(nn.Module):
            def __init__(self):
                super().__init__()
                self.lifting = nn.Linear(n_channels_in, width)
                self.spectral = nn.ModuleList(
                    [_SpectralConv1d.build(width, width, n_modes) for _ in range(n_layers)]
                )
                self.pointwise = nn.ModuleList(
                    [nn.Conv1d(width, width, 1) for _ in range(n_layers)]
                )
                self.projection = nn.Sequential(
                    nn.Linear(width, 128), nn.GELU(), nn.Linear(128, 1)
                )

            def forward(self, x):
                # x: (B, L, C_in)  -> lifting to (B, L, width) -> (B, width, L)
                x = self.lifting(x).transpose(1, 2)
                for sc, pc in zip(self.spectral, self.pointwise):
                    x = sc(x) + pc(x)
                    x = torch.nn.functional.gelu(x)
                x = x.transpose(1, 2)  # (B, L, width)
                return self.projection(x).squeeze(-1)  # (B, L)

        return FNO1d()


class FNOSurrogate(SurrogateModel):
    """FNO-1d surrogate along the t-axis, with MLP fallback."""

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.mode: str = "unknown"  # "fno1d" or "mlp_fallback"
        self.model = None
        self._t_grid: np.ndarray | None = None
        self._input_dim: int | None = None

    def name(self) -> str:
        return "fno"

    def supports_uq(self) -> bool:
        return False

    # ------------------------------------------------------------------ fit
    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        _require_torch()
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        X_train = self._as_2d(X_train)
        y_train = self._as_1d(y_train)
        self._input_dim = X_train.shape[1]

        is_grid, info = _detect_grid(X_train)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if not is_grid:
            logger.warning(
                "FNO: input data is not gridded on t — falling back to ResNet MLP. "
                "Provide data with a shared t-grid per (Q2,W2) slice to enable true FNO."
            )
            self.mode = "mlp_fallback"
            self.model = _MLPCore.build(
                input_dim=self._input_dim,
                hidden_layers=[self.cfg.width, self.cfg.width, self.cfg.width],
                activation="gelu",
                use_resnet=True,
            ).to(device)
            opt = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr)
            loss_fn = torch.nn.MSELoss()
            ds = TensorDataset(
                torch.from_numpy(X_train).float(),
                torch.from_numpy(y_train.reshape(-1, 1)).float(),
            )
            loader = DataLoader(ds, batch_size=max(32, self.cfg.batch_size), shuffle=True)
            for epoch in range(self.cfg.max_epochs):
                self.model.train()
                for xb, yb in loader:
                    xb = xb.to(device)
                    yb = yb.to(device)
                    opt.zero_grad()
                    loss = loss_fn(self.model(xb), yb)
                    loss.backward()
                    opt.step()
            self.model.eval()
            return

        # --- True FNO-1d path ---
        self.mode = "fno1d"
        t_grid = info["t_grid"]  # (L,)
        L = t_grid.size
        self._t_grid = t_grid

        # Build per-slice samples: (Q2, W2) slices → (L,) targets.
        q = np.round(X_train[:, 0], decimals=8)
        w = np.round(X_train[:, 1], decimals=8)
        keys_uniq, inv = np.unique(np.stack([q, w], axis=1), axis=0, return_inverse=True)
        n_slices = keys_uniq.shape[0]

        # Per-slice target: reorder by t.
        Y_grid = np.zeros((n_slices, L), dtype=np.float64)
        for s in range(n_slices):
            idx = np.where(inv == s)[0]
            order = np.argsort(X_train[idx, 2])
            Y_grid[s] = y_train[idx[order]]

        # Per-slice input tensor: 3 channels (Q2, W2, t_grid) broadcast over L.
        X_grid = np.zeros((n_slices, L, 3), dtype=np.float64)
        X_grid[:, :, 0] = keys_uniq[:, 0:1]
        X_grid[:, :, 1] = keys_uniq[:, 1:2]
        X_grid[:, :, 2] = t_grid[None, :]

        self.model = _FNO1d.build(
            n_channels_in=3,
            width=self.cfg.width,
            n_modes=self.cfg.n_modes,
            n_layers=self.cfg.n_layers,
        ).to(device)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr)
        loss_fn = torch.nn.MSELoss()
        batch = max(1, min(self.cfg.batch_size, n_slices))
        ds = TensorDataset(
            torch.from_numpy(X_grid).float(), torch.from_numpy(Y_grid).float()
        )
        loader = torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=True)
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
            if epoch % 50 == 0:
                logger.debug("FNO epoch %d  loss=%.4e", epoch, float(loss.detach().cpu()))
        self.model.eval()

    # -------------------------------------------------------------- predict
    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        if self.model is None:
            raise RuntimeError("FNO.predict called before fit/load")
        import torch

        X = self._as_2d(X)
        device = next(self.model.parameters()).device

        if self.mode == "mlp_fallback":
            self.model.eval()
            with torch.no_grad():
                out = self.model(torch.from_numpy(X).float().to(device)).cpu().numpy().ravel()
            return out, None

        # FNO path: assemble a (B, L, 3) tensor per unique slice, predict, then
        # scatter back by nearest-neighbor t-lookup.
        assert self._t_grid is not None
        t_grid = self._t_grid
        L = t_grid.size
        q = np.round(X[:, 0], decimals=8)
        w = np.round(X[:, 1], decimals=8)
        keys = np.stack([q, w], axis=1)
        keys_uniq, inv = np.unique(keys, axis=0, return_inverse=True)
        n_slices = keys_uniq.shape[0]
        X_grid = np.zeros((n_slices, L, 3), dtype=np.float64)
        X_grid[:, :, 0] = keys_uniq[:, 0:1]
        X_grid[:, :, 1] = keys_uniq[:, 1:2]
        X_grid[:, :, 2] = t_grid[None, :]

        self.model.eval()
        with torch.no_grad():
            grid_pred = self.model(torch.from_numpy(X_grid).float().to(device)).cpu().numpy()

        # For each input row, find its nearest t-gridpoint.
        out = np.empty(X.shape[0], dtype=np.float64)
        for i in range(X.shape[0]):
            s = inv[i]
            j = int(np.argmin(np.abs(t_grid - X[i, 2])))
            out[i] = grid_pred[s, j]
        return out, None

    # ---------------------------------------------------------------- save
    def save(self, path: Path) -> None:
        if self.model is None:
            raise RuntimeError("FNO.save called before fit")
        import torch

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path / "weights.pt")
        meta = {
            "mode": self.mode,
            "input_dim": self._input_dim,
            "width": self.cfg.width,
            "n_modes": self.cfg.n_modes,
            "n_layers": self.cfg.n_layers,
            "t_grid": self._t_grid.tolist() if self._t_grid is not None else None,
        }
        with (path / "fno_meta.json").open("w") as f:
            json.dump(meta, f)

    # ---------------------------------------------------------------- load
    def load(self, path: Path) -> None:
        _require_torch()
        import torch

        path = Path(path)
        with (path / "fno_meta.json").open() as f:
            meta = json.load(f)
        self.mode = meta["mode"]
        self._input_dim = meta["input_dim"]
        self._t_grid = np.asarray(meta["t_grid"]) if meta["t_grid"] is not None else None
        if self.mode == "mlp_fallback":
            self.model = _MLPCore.build(
                input_dim=self._input_dim,
                hidden_layers=[meta["width"], meta["width"], meta["width"]],
                activation="gelu",
                use_resnet=True,
            )
        else:
            self.model = _FNO1d.build(
                n_channels_in=3,
                width=meta["width"],
                n_modes=meta["n_modes"],
                n_layers=meta["n_layers"],
            )
        self.model.load_state_dict(torch.load(path / "weights.pt", map_location="cpu"))
        self.model.eval()
