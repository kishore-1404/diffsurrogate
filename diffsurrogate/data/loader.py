"""Dataset loader.

Supports three on-disk formats:
  - CSV   (.csv)         via pandas
  - HDF5  (.h5/.hdf5)    via pandas (requires pytables or h5py)
  - NumPy (.npy)         structured array with named fields

Returns a DataFrame plus numpy arrays ``X`` of shape (N, 3) and ``y`` of shape
(N, 1), ordered by the ``t`` column to make t-spectrum plots easier downstream.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def load_dataset(
    path: str | Path,
    input_columns: list[str],
    target_column: str,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Load a lookup table from disk.

    Parameters
    ----------
    path : str | Path
        Path to the data file. Extension determines format.
    input_columns : list[str]
        Names of the three kinematic columns (Q2, W2, t) in the chosen schema.
    target_column : str
        Name of the target column (e.g. ``"logA"``).

    Returns
    -------
    df : DataFrame
        Full dataframe, sorted by the t-column. All columns preserved.
    X : ndarray of shape (N, 3)
        Kinematic inputs in the order given by ``input_columns``.
    y : ndarray of shape (N, 1)
        Target values as a column vector.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    ext = path.suffix.lower()
    if ext == ".csv":
        df = pd.read_csv(path)
    elif ext in {".h5", ".hdf5"}:
        # Try the default key; if unique, pandas handles it automatically.
        try:
            df = pd.read_hdf(path)  # type: ignore[assignment]
            if isinstance(df, pd.Series):
                df = df.to_frame()
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"Failed to read HDF5 {path}: {e}") from e
    elif ext == ".npy":
        arr = np.load(path, allow_pickle=False)
        if arr.dtype.names is None:
            raise ValueError(
                f".npy file {path} is not a structured array; "
                f"cannot interpret columns"
            )
        df = pd.DataFrame(arr)
    else:
        raise ValueError(f"Unsupported file extension '{ext}' for {path}")

    missing = [c for c in input_columns + [target_column] if c not in df.columns]
    if missing:
        raise KeyError(
            f"Columns missing from {path}: {missing}. "
            f"Available: {list(df.columns)}"
        )

    # Sort by the t column (3rd input) for stable t-spectrum plots.
    t_col = input_columns[2]
    df = df.sort_values(t_col).reset_index(drop=True)

    X = df[input_columns].to_numpy(dtype=np.float64)
    y = df[target_column].to_numpy(dtype=np.float64).reshape(-1, 1)

    if np.isnan(X).any() or np.isnan(y).any():
        n_nan = int(np.isnan(X).any(axis=1).sum() + np.isnan(y).any(axis=1).sum())
        logger.warning("Dataset contains %d rows with NaN; these will pass through.", n_nan)

    logger.info("Loaded %d samples from %s", len(df), path)
    return df, X, y
