"""Abstract base class for all surrogate models.

Every concrete model implements fit / predict / save / load / supports_uq /
name, working entirely in *scaled* space. Wrapping scalers around the model
is the orchestrator's job (benchmark/train/predict commands), not the model's.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


class SurrogateModel(ABC):
    """Abstract surrogate model.

    Contract:
      - ``fit`` and ``predict`` both operate on *scaled* inputs and targets.
      - ``predict`` returns ``(mean, std)`` where ``std`` is ``None`` if the
        model cannot quantify uncertainty.
      - ``save`` and ``load`` persist everything needed to reproduce predict
        bitwise-identically (or as close as the framework allows).
    """

    @abstractmethod
    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Fit the model on scaled training data.

        Parameters
        ----------
        X_train : (N, D) array in scaled space
        y_train : (N, 1) or (N,) array in scaled space
        """

    @abstractmethod
    def predict(
        self, X: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Predict on scaled inputs.

        Returns
        -------
        mean : (N,) ndarray in scaled space
        std  : (N,) ndarray in scaled space or ``None`` if UQ not supported
        """

    @abstractmethod
    def supports_uq(self) -> bool:
        """True iff ``predict`` returns a non-None second element."""

    @abstractmethod
    def save(self, path: Path) -> None:
        """Save all state needed to reload this model to ``path`` (a directory)."""

    @abstractmethod
    def load(self, path: Path) -> None:
        """Restore state from a directory produced by ``save``."""

    @abstractmethod
    def name(self) -> str:
        """Short, stable string identifier used in registry and filenames."""

    # ---- tiny convenience helpers shared by subclasses ----

    @staticmethod
    def _as_2d(X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return X

    @staticmethod
    def _as_1d(y: np.ndarray) -> np.ndarray:
        return np.asarray(y, dtype=np.float64).ravel()
