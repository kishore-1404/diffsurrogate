"""Polynomial Chaos Expansion surrogate via ``chaospy``.

We assume the scaled inputs are approximately standard-normal after our
transforms, so we use Hermite polynomials (``chaospy.J(*Normal(0,1))``).
This matches the spirit of the paper (orthogonal polynomials w.r.t. the
input density) while keeping the basis family fixed regardless of the
raw kinematic distribution.

For uncertainty quantification we report the *residual standard deviation*
on training data as the predictive sigma. This is a lightweight, reasonable
proxy that (a) is cheap, (b) doesn't rely on per-point structure, and (c)
produces a well-defined sigma for coverage/NLL metrics. A full bootstrap
over the regression coefficients would give per-point sigmas and is a
natural extension.
"""

from __future__ import annotations

import json
import logging
import pickle
import warnings
from pathlib import Path

import numpy as np

from diffsurrogate.models.base import SurrogateModel

logger = logging.getLogger(__name__)


def _require_chaospy():
    try:
        import chaospy  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "PCESurrogate requires chaospy. Install with: "
            "pip install diffsurrogate[chaospy]"
        ) from e


def _regression_estimator(name: str):
    """Map our config string to a scikit-learn estimator for chaospy.

    chaospy's fit_regression expects an sklearn BaseEstimator or None.
    We disable fit_intercept because the polynomial basis already includes
    a constant term.
    """
    from sklearn.linear_model import ElasticNet, Lars, Lasso, LinearRegression

    name = name.lower()
    if name == "ols":
        return LinearRegression(fit_intercept=False)
    if name == "lars":
        # Lars / LassoLars give the LARS-style sparse fit the spec asks for.
        return Lars(fit_intercept=False, n_nonzero_coefs=500)
    if name == "lasso":
        return Lasso(fit_intercept=False, alpha=1e-4, max_iter=10000)
    if name == "elastic_net":
        return ElasticNet(fit_intercept=False, alpha=1e-4, max_iter=10000)
    raise ValueError(f"Unknown PCE regression '{name}'")


class PCESurrogate(SurrogateModel):
    """PCE surrogate via chaospy."""

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self._expansion = None
        self._surrogate = None
        self._residual_std: float | None = None
        self._input_dim: int | None = None

    def name(self) -> str:
        return "pce"

    def supports_uq(self) -> bool:
        return True

    # ------------------------------------------------------------------ fit
    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        _require_chaospy()
        import chaospy as cp

        X_train = self._as_2d(X_train)
        y_train = self._as_1d(y_train)
        self._input_dim = X_train.shape[1]

        if self.cfg.order > 5:
            warnings.warn(
                f"PCE order={self.cfg.order} > 5: high risk of Runge-phenomenon "
                "oscillations near boundaries.",
                UserWarning,
                stacklevel=2,
            )

        # Standard-normal marginals on scaled inputs: robust default since our
        # transforms approximately z-score each axis.
        marginals = [cp.Normal(0.0, 1.0) for _ in range(self._input_dim)]
        joint = cp.J(*marginals)
        self._expansion = cp.generate_expansion(self.cfg.order, joint)

        # chaospy expects abscissas with shape (D, K).
        nodes = X_train.T
        try:
            estimator = _regression_estimator(self.cfg.regression)
            self._surrogate = cp.fit_regression(
                self._expansion, nodes, y_train, model=estimator
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "PCE regression='%s' failed (%s); falling back to default OLS.",
                self.cfg.regression, e,
            )
            self._surrogate = cp.fit_regression(self._expansion, nodes, y_train)

        # Residual std on training data → predictive sigma.
        y_pred = self._eval_surrogate(X_train)
        resid = y_train - y_pred
        if resid.size > 1:
            self._residual_std = float(np.std(resid, ddof=1))
        else:
            self._residual_std = float(np.std(resid))
        if not np.isfinite(self._residual_std) or self._residual_std <= 0:
            self._residual_std = 1e-3
        logger.info(
            "PCE fitted (order=%d, reg=%s, residual_std=%.4e)",
            self.cfg.order, self.cfg.regression, self._residual_std,
        )

    def _eval_surrogate(self, X: np.ndarray) -> np.ndarray:
        """Evaluate the fitted polynomial at rows of ``X`` (shape (N, D))."""
        X = np.asarray(X, dtype=np.float64)
        out = self._surrogate(*[X[:, i] for i in range(X.shape[1])])
        return np.asarray(out, dtype=np.float64).ravel()

    # -------------------------------------------------------------- predict
    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        if self._surrogate is None:
            raise RuntimeError("PCE.predict called before fit/load")
        X = self._as_2d(X)
        mean = self._eval_surrogate(X)
        std = np.full_like(mean, self._residual_std or 0.0)
        return mean, std

    # ---------------------------------------------------------------- save
    def save(self, path: Path) -> None:
        if self._surrogate is None:
            raise RuntimeError("PCE.save called before fit")
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        with (path / "pce_model.pkl").open("wb") as f:
            pickle.dump(
                {
                    "expansion": self._expansion,
                    "surrogate": self._surrogate,
                    "residual_std": self._residual_std,
                },
                f,
            )
        with (path / "pce_meta.json").open("w") as f:
            json.dump({"input_dim": self._input_dim, "order": self.cfg.order}, f)

    # ---------------------------------------------------------------- load
    def load(self, path: Path) -> None:
        _require_chaospy()
        path = Path(path)
        with (path / "pce_meta.json").open() as f:
            meta = json.load(f)
        self._input_dim = meta["input_dim"]
        with (path / "pce_model.pkl").open("rb") as f:
            bundle = pickle.load(f)  # noqa: S301 — trusted local artifact
        self._expansion = bundle["expansion"]
        self._surrogate = bundle["surrogate"]
        self._residual_std = bundle["residual_std"]
