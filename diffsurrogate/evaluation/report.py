"""Benchmark report writer.

Given a list of per-model result dicts, writes a CSV and a JSON file into
``output_dir`` and emits a human-readable leaderboard to stdout.

A results row looks like::

    {
        "model": "neural_net",
        "rmse": 0.123,
        "nll": nan,
        "coverage_95": nan,
        "constraint_violation_rate": nan,
        "fit_time_sec": 2.3,
        "predict_time_sec": 0.01,
        "n_train": 800,
        "n_test": 200,
        "supports_uq": false,
    }
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def write_report(
    results: list[dict[str, Any]],
    output_dir: Path | str,
    metrics: list[str],
) -> tuple[Path, Path]:
    """Write ``benchmark_results.{csv,json}`` into ``output_dir``.

    Returns the pair of paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Stable column order: model, metrics in config order, then bookkeeping.
    columns = ["model"] + list(metrics) + [
        "fit_time_sec", "predict_time_sec", "n_train", "n_test", "supports_uq",
    ]
    rows = []
    for r in results:
        row = {c: r.get(c, None) for c in columns}
        rows.append(row)
    df = pd.DataFrame(rows, columns=columns)

    csv_path = output_dir / "benchmark_results.csv"
    json_path = output_dir / "benchmark_results.json"
    df.to_csv(csv_path, index=False)
    with json_path.open("w") as f:
        # Replace NaN with None for valid JSON.
        safe_rows = [
            {k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in r.items()}
            for r in rows
        ]
        json.dump({"metrics": metrics, "rows": safe_rows}, f, indent=2, default=str)

    logger.info("Wrote benchmark report: %s, %s", csv_path, json_path)
    return csv_path, json_path


def print_leaderboard(results: list[dict[str, Any]], metrics: list[str]) -> None:
    """Print an ASCII leaderboard ranked by RMSE (lower=better)."""
    if not results:
        print("(no results)")
        return

    sorted_rows = sorted(
        results,
        key=lambda r: r.get("rmse", float("inf")) if r.get("rmse") is not None else float("inf"),
    )

    cols = ["model"] + metrics + ["fit_time_sec"]
    widths = {c: max(len(c), 6) for c in cols}
    for r in sorted_rows:
        for c in cols:
            widths[c] = max(widths[c], len(_fmt(r.get(c))))

    # header
    sep = "+" + "+".join("-" * (widths[c] + 2) for c in cols) + "+"
    header = "|" + "|".join(f" {c:<{widths[c]}} " for c in cols) + "|"
    print(sep)
    print(header)
    print(sep)
    for rank, r in enumerate(sorted_rows, start=1):
        cells = []
        for c in cols:
            v = _fmt(r.get(c))
            cells.append(f" {v:<{widths[c]}} ")
        prefix = f"#{rank} "
        print("|" + "|".join(cells) + "|")
    print(sep)


def _fmt(v: Any) -> str:
    if v is None:
        return "–"
    if isinstance(v, float):
        if math.isnan(v):
            return "nan"
        if abs(v) < 1e-3 or abs(v) >= 1e5:
            return f"{v:.3e}"
        return f"{v:.4f}"
    if isinstance(v, bool):
        return "yes" if v else "no"
    return str(v)
