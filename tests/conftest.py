"""Shared pytest fixtures for the diffsurrogate test suite."""

from __future__ import annotations

import numpy as np
import pytest


def _synthetic_diffractive(n: int = 200, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Generate (X, y) from a toy diffractive function.

    X columns: (Q2_log_center, W2_center, t_center) — matches the package schema.
    y = logA where A has exp(-|t|) * sin(t) structure plus mild Q²/W² modulation.
    """
    rng = np.random.default_rng(seed)
    # Spread Q2 and W2 across a realistic range.
    q2 = rng.uniform(1.0, 10.0, size=n)
    w2 = rng.uniform(20.0, 200.0, size=n)
    t = -rng.uniform(0.02, 1.5, size=n)

    log_q2 = np.log(q2)
    abs_t = np.abs(t)
    # exp(-B|t|) * sin(t)^2 pattern gives dip-like structure.
    A2 = np.exp(-4.0 * abs_t) * (np.sin(2.0 * np.pi * abs_t) ** 2 + 1e-3)
    A2 *= np.exp(-0.05 * log_q2)
    A2 *= 1.0 + 0.02 * np.log(w2 / 10.0)
    logA = np.log(A2) + rng.normal(0.0, 0.01, size=n)

    X = np.stack([log_q2, w2, t], axis=1)
    y = logA.reshape(-1, 1)
    return X.astype(np.float64), y.astype(np.float64)


@pytest.fixture
def synthetic_dataset() -> tuple[np.ndarray, np.ndarray]:
    """200-point synthetic diffractive dataset, deterministic."""
    return _synthetic_diffractive(n=200, seed=0)


@pytest.fixture
def small_dataset() -> tuple[np.ndarray, np.ndarray]:
    """100-point set for smoke-testing fits quickly."""
    return _synthetic_diffractive(n=100, seed=1)


@pytest.fixture
def predict_inputs() -> np.ndarray:
    """20-point prediction input set."""
    X, _ = _synthetic_diffractive(n=20, seed=99)
    return X
