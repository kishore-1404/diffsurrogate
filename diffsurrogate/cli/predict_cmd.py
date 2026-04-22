"""Predict subcommand — Mode 2b.

For each enabled (or --models overridden) surrogate, loads its production
artifact (model + scalers), applies the loaded scalers to new inputs,
predicts, inverse-transforms back to physical amplitude units, and writes
``predictions_{model}.csv`` under the evaluation output directory.

UQ-supporting models additionally write ``std`` and
``lower_2sigma`` / ``upper_2sigma`` columns in physical units.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from diffsurrogate.config import Config
from diffsurrogate.persistence.loader import load_model

logger = logging.getLogger(__name__)


def run(cfg: Config, model_names: list[str] | None = None) -> int:
    predict_path = Path(cfg.data.predict_path)
    if not predict_path.exists():
        logger.error("predict_path does not exist: %s", predict_path)
        return 2

    df_in = pd.read_csv(predict_path)
    missing = [c for c in cfg.data.input_columns if c not in df_in.columns]
    if missing:
        logger.error("predict CSV %s is missing columns: %s", predict_path, missing)
        return 2
    X_raw = df_in[cfg.data.input_columns].to_numpy(dtype=np.float64)

    names = model_names if model_names else cfg.models.enabled
    model_dir = Path(cfg.persistence.model_dir)
    output_dir = Path(cfg.evaluation.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []
    for name in names:
        save_dir = model_dir / name / "production"
        if not save_dir.exists():
            logger.warning("No production artifact for '%s' at %s — skipping", name, save_dir)
            summary_rows.append({"model": name, "status": "no artifact", "output": "—"})
            continue
        sub_cfg = cfg.models.get(name)

        try:
            model, scalers = load_model(name, save_dir, model_config=sub_cfg)
        except ImportError as e:
            logger.warning("'%s' skipped (missing dep): %s", name, e)
            summary_rows.append({"model": name, "status": f"missing dep: {e}", "output": "—"})
            continue
        except Exception as e:  # noqa: BLE001
            logger.exception("load_model('%s') failed: %s", name, e)
            summary_rows.append({"model": name, "status": f"load failed: {e}", "output": "—"})
            continue

        input_scaler = scalers.get("input_scaler")
        target_scaler = scalers.get("target_scaler")
        if input_scaler is None or target_scaler is None:
            logger.warning("'%s' has no saved scalers — predictions will be in scaled space", name)

        # Apply loaded scalers (no refitting).
        X_s = input_scaler.transform(X_raw) if input_scaler is not None else X_raw

        mu_s, std_s = model.predict(X_s)

        # Inverse-transform to physical units. Our target scaler inverse
        # returns ln(A); we exp to get A.
        if target_scaler is not None:
            mu_lnA = target_scaler.inverse_transform(mu_s)
        else:
            mu_lnA = mu_s
        with np.errstate(over="ignore"):
            mu_phys = np.exp(np.asarray(mu_lnA).ravel())

        out_df = df_in.copy()
        out_df["predicted_ln_amplitude"] = np.asarray(mu_lnA).ravel()
        out_df["predicted_amplitude"] = mu_phys

        if std_s is not None:
            # Convert std from scaled to ln(A) space.
            if target_scaler is not None and hasattr(target_scaler, "std_to_original"):
                std_lnA = np.asarray(target_scaler.std_to_original(std_s)).ravel()
            else:
                std_lnA = np.asarray(std_s).ravel()
            # Bands on ln(A), then exp → multiplicative in A.
            lo_lnA = np.asarray(mu_lnA).ravel() - 2.0 * std_lnA
            hi_lnA = np.asarray(mu_lnA).ravel() + 2.0 * std_lnA
            with np.errstate(over="ignore"):
                lo_phys = np.exp(lo_lnA)
                hi_phys = np.exp(hi_lnA)
            out_df["std_ln_amplitude"] = std_lnA
            out_df["lower_2sigma"] = lo_phys
            out_df["upper_2sigma"] = hi_phys

        out_path = output_dir / f"predictions_{name}.csv"
        out_df.to_csv(out_path, index=False)
        summary_rows.append({
            "model": name,
            "status": "ok",
            "n_points": int(X_raw.shape[0]),
            "uq": bool(std_s is not None),
            "output": str(out_path),
        })
        logger.info("Wrote %s (%d points)", out_path, X_raw.shape[0])

    _print_summary(summary_rows)
    return 0


def _print_summary(rows: list[dict[str, Any]]) -> None:
    if not rows:
        print("(no models predicted)")
        return
    print()
    print("Predict summary")
    print("-" * 60)
    for r in rows:
        status = r.get("status", "?")
        model = r.get("model", "?")
        output = r.get("output", "—")
        extra = ""
        if r.get("n_points") is not None:
            extra = f"  N={r['n_points']}, UQ={'yes' if r.get('uq') else 'no'}"
        print(f"  [{status:10}] {model:18} → {output}{extra}")
    print()
