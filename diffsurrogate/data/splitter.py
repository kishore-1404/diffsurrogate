"""Train/test splitter that stratifies on decile bins of ``|t|``.

Naïve random splitting can leave the rare diffractive-dip points (extreme ``|t|``)
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
    """Split (X, y) into train/test sets, stratified on decile bins of |t|.

    Parameters
    ----------
    X : ndarray (N, D)
        Input features. The ``t_column_index``-th column is the ``t`` axis.
    y : ndarray (N, 1) or (N,)
        Targets.
    train_fraction : float
        Fraction of data to put in the train set.
    random_seed : int
        Seed for the per-bin shuffle.
    n_bins : int
        Number of quantile bins (default 10 = deciles).
    t_column_index : int
        Which column of X to stratify on.

    Returns
    -------
    X_train, X_test, y_train, y_test : ndarray
    """
    X = np.asarray(X)
    y = np.asarray(y)
    n = X.shape[0]
    if n < 2 * n_bins:
        logger.warning(
            "N=%d is small; falling back to plain random split (n_bins=%d requested)",
            n,
            n_bins,
        )
        return _random_split(X, y, train_fraction, random_seed)

    rng = np.random.default_rng(random_seed)
    t_abs = np.abs(X[:, t_column_index])

    # Quantile edges; use np.quantile for stability against ties.
    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.quantile(t_abs, qs)
    # Expand the outer edges slightly so everything falls inside.
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
        # Guarantee at least 1 sample in test too, when the bin has >= 2.
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
    return X[train_idx_arr], X[test_idx_arr], y[train_idx_arr], y[test_idx_arr]


def _random_split(
    X: np.ndarray, y: np.ndarray, train_fraction: float, random_seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_seed)
    n = X.shape[0]
    perm = rng.permutation(n)
    n_train = max(1, int(round(train_fraction * n)))
    n_train = min(n_train, n - 1)
    tr = perm[:n_train]
    te = perm[n_train:]
    return X[tr], X[te], y[tr], y[te]
