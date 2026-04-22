"""Train subcommand — Mode 2a.

Fits scalers on the full dataset (no test split), trains every enabled
model, and writes each to ``model_dir/{name}/production/``. Emits a
confirmation table with file-size and timestamp.
"""

from __future__ import annotations

import logging
import time
import warnings
from dataclasses import asdict
from pathlib import Path

from diffsurrogate.config import Config
from diffsurrogate.data import build_input_scaler, build_target_scaler, load_dataset
from diffsurrogate.models import build_model
from diffsurrogate.persistence.saver import artifact_size_bytes, save_model

logger = logging.getLogger(__name__)


def run(cfg: Config) -> int:
    _, X, y = load_dataset(
        cfg.data.input_path,
        cfg.data.input_columns,
        cfg.data.target_column,
    )

    input_scaler = build_input_scaler(cfg.transforms, cfg.data)
    target_scaler = build_target_scaler(cfg.transforms)
    X_s = input_scaler.fit_transform(X)
    y_s = target_scaler.fit_transform(y).ravel()

    model_dir = Path(cfg.persistence.model_dir)
    rows: list[dict] = []

    for name in cfg.models.enabled:
        sub_cfg = cfg.models.get(name)
        if sub_cfg is None:
            logger.warning("'%s' is enabled but has no [models.%s] sub-table — skipping", name, name)
            continue

        logger.info("Training %s on full dataset (N=%d)…", name, X_s.shape[0])
        try:
            model = build_model(name, sub_cfg)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                t0 = time.time()
                model.fit(X_s, y_s)
                elapsed = time.time() - t0

            save_dir = model_dir / name / "production"
            save_model(
                model,
                {"input_scaler": input_scaler, "target_scaler": target_scaler},
                save_dir,
                n_train=int(X_s.shape[0]),
                config_snapshot={
                    "model": asdict(sub_cfg) if hasattr(sub_cfg, "__dataclass_fields__") else {},
                    "transforms": asdict(cfg.transforms),
                    "data": asdict(cfg.data),
                },
            )
            rows.append({
                "model": name,
                "path": str(save_dir.resolve()),
                "size_bytes": artifact_size_bytes(save_dir),
                "fit_time_sec": round(elapsed, 2),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            })
        except ImportError as e:
            logger.warning("'%s' skipped (missing dependency): %s", name, e)
            rows.append({"model": name, "path": "—", "size_bytes": 0, "fit_time_sec": None,
                         "timestamp": "skipped: missing dep"})
        except Exception as e:  # noqa: BLE001
            logger.exception("Training %s failed: %s", name, e)
            rows.append({"model": name, "path": "—", "size_bytes": 0, "fit_time_sec": None,
                         "timestamp": f"failed: {e}"})

    _print_table(rows)
    return 0


def _print_table(rows: list[dict]) -> None:
    if not rows:
        print("(no models trained)")
        return
    cols = ["model", "path", "size_bytes", "fit_time_sec", "timestamp"]
    widths = {c: max(len(c), 4) for c in cols}
    for r in rows:
        for c in cols:
            widths[c] = max(widths[c], len(str(r.get(c, ""))))
    sep = "+" + "+".join("-" * (widths[c] + 2) for c in cols) + "+"
    print(sep)
    print("|" + "|".join(f" {c:<{widths[c]}} " for c in cols) + "|")
    print(sep)
    for r in rows:
        print("|" + "|".join(f" {str(r.get(c, '')):<{widths[c]}} " for c in cols) + "|")
    print(sep)
