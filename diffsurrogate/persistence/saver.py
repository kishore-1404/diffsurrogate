"""Atomic model saver.

``save_model`` writes the model's internal state (whatever its ``save()``
method produces), the fitted scalers, and a ``metadata.json`` sidecar to a
target directory. The writes go into a ``.partial`` sibling directory first
and are renamed into place on success so that partial/crashed writes don't
corrupt a previously-saved bundle.

Typical layout on disk:

    saved_models/
      neural_net/
        production/
          nn_meta.json
          weights.pt
          scalers.joblib
          metadata.json
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Any

import joblib

from diffsurrogate.models.base import SurrogateModel

logger = logging.getLogger(__name__)


def _dir_size_bytes(path: Path) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            fp = Path(root) / f
            try:
                total += fp.stat().st_size
            except OSError:
                pass
    return total


def save_model(
    model: SurrogateModel,
    scalers: dict[str, Any] | None,
    base_dir: Path | str,
    n_train: int | None = None,
    config_snapshot: dict | None = None,
) -> Path:
    """Save a fitted model + its scalers to ``base_dir`` atomically.

    Parameters
    ----------
    model : SurrogateModel
        A fitted model. Its ``save()`` method is called to persist weights.
    scalers : dict | None
        Dict of fitted scaler objects (joblib-pickleable). Typical keys:
        ``"input_scaler"``, ``"target_scaler"``.
    base_dir : Path
        Final directory. Will be created. If it exists, contents are
        replaced (after the new bundle is successfully written).
    n_train : int | None
        Training set size, stored in ``metadata.json`` for traceability.
    config_snapshot : dict | None
        Arbitrary JSON-serializable snapshot of the config used for this
        training run.

    Returns
    -------
    The final ``base_dir`` (as a resolved ``Path``).
    """
    base_dir = Path(base_dir)
    partial_dir = base_dir.with_name(base_dir.name + ".partial")
    if partial_dir.exists():
        shutil.rmtree(partial_dir)
    partial_dir.mkdir(parents=True, exist_ok=True)

    try:
        # 1. Let the model write its weights / artifacts into partial_dir.
        model.save(partial_dir)

        # 2. Scalers.
        if scalers is not None:
            joblib.dump(scalers, partial_dir / "scalers.joblib")

        # 3. Metadata.
        meta = {
            "model_name": model.name(),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z") or time.strftime("%Y-%m-%dT%H:%M:%S"),
            "n_train": int(n_train) if n_train is not None else None,
            "supports_uq": bool(model.supports_uq()),
            "config_snapshot": config_snapshot or {},
        }
        with (partial_dir / "metadata.json").open("w") as f:
            json.dump(meta, f, indent=2, default=str)

        # 4. Atomic swap.
        if base_dir.exists():
            backup = base_dir.with_name(base_dir.name + ".backup")
            if backup.exists():
                shutil.rmtree(backup)
            base_dir.rename(backup)
            try:
                partial_dir.rename(base_dir)
                shutil.rmtree(backup)
            except Exception:
                # Roll back.
                if base_dir.exists():
                    shutil.rmtree(base_dir)
                backup.rename(base_dir)
                raise
        else:
            partial_dir.rename(base_dir)
    except Exception:
        if partial_dir.exists():
            shutil.rmtree(partial_dir, ignore_errors=True)
        raise

    size = _dir_size_bytes(base_dir)
    logger.info("Saved %s → %s (%d bytes)", model.name(), base_dir, size)
    return base_dir


def artifact_size_bytes(path: Path | str) -> int:
    """Return total size in bytes of a saved-model directory (for reporting)."""
    p = Path(path)
    if not p.exists():
        return 0
    return _dir_size_bytes(p)
