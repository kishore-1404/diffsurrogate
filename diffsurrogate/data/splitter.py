"""Train/test splitter that stratifies on decile bins of ``|t|``.

Naive random splitting can leave the rare diffractive-dip points (extreme ``|t|``)
entirely in one split, which then ruins out-of-sample evaluation because the
test set never sees the physics that matters most. Stratifying on deciles of
``|t|`` keeps the tails represented in both splits.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def stratified_split_on_t(
    X: np.ndarray,
    y: np.ndarray,
    train_fraction: float,
    random_seed: int,
    n_bins: int = 10,
    t_column_index: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split ``(X, y)`` into train/test arrays, stratified on decile bins of ``|t|``."""
    train_idx, test_idx = stratified_split_indices_on_t(
        X,
        train_fraction=train_fraction,
        random_seed=random_seed,
        n_bins=n_bins,
        t_column_index=t_column_index,
    )
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def stratified_split_indices_on_t(
    X: np.ndarray,
    train_fraction: float,
    random_seed: int,
    n_bins: int = 10,
    t_column_index: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Return train/test indices stratified on quantile bins of ``|t|``."""
    X = np.asarray(X)
    n = X.shape[0]
    if n < 2 * n_bins:
        logger.warning(
            "N=%d is small; falling back to plain random split (n_bins=%d requested)",
            n,
            n_bins,
        )
        return _random_split_indices(n, train_fraction, random_seed)

    rng = np.random.default_rng(random_seed)
    t_abs = np.abs(X[:, t_column_index])

    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.quantile(t_abs, qs)
    edges[0] -= 1e-12
    edges[-1] += 1e-12
    bin_ids = np.clip(np.digitize(t_abs, edges[1:-1], right=False), 0, n_bins - 1)

    train_idx: list[int] = []
    test_idx: list[int] = []
    for b in range(n_bins):
        idx = np.where(bin_ids == b)[0]
        if idx.size == 0:
            continue
        rng.shuffle(idx)
        n_train = max(1, int(round(train_fraction * idx.size)))
        if idx.size >= 2 and n_train == idx.size:
            n_train = idx.size - 1
        train_idx.extend(idx[:n_train].tolist())
        test_idx.extend(idx[n_train:].tolist())

    train_idx_arr = np.array(sorted(train_idx), dtype=np.int64)
    test_idx_arr = np.array(sorted(test_idx), dtype=np.int64)

    logger.info(
        "Stratified split on |t| deciles: %d train, %d test (train_fraction=%.2f)",
        train_idx_arr.size,
        test_idx_arr.size,
        train_fraction,
    )
    return train_idx_arr, test_idx_arr


def _random_split(
    X: np.ndarray, y: np.ndarray, train_fraction: float, random_seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    tr, te = _random_split_indices(X.shape[0], train_fraction, random_seed)
    return X[tr], X[te], y[tr], y[te]


def _random_split_indices(n: int, train_fraction: float, random_seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_seed)
    perm = rng.permutation(n)
    n_train = max(1, int(round(train_fraction * n)))
    n_train = min(n_train, n - 1)
    tr = np.sort(perm[:n_train])
    te = np.sort(perm[n_train:])
    return tr, te
