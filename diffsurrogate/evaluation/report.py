"""Benchmark report writer."""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

import pandas as pd

from diffsurrogate.evaluation.research import method_rows

logger = logging.getLogger(__name__)

_STANDARD_RESULT_COLUMNS = [
    "model",
    "status",
    "error",
    "fit_time_sec",
    "predict_time_sec",
    "n_train",
    "n_test",
    "supports_uq",
    "artifact_size_bytes",
    "artifact_dir",
    "predictions_path",
    "residual_mean",
    "residual_std",
    "abs_residual_p50",
    "abs_residual_p90",
    "max_abs_residual",
]


def write_report(
    results: list[dict[str, Any]],
    output_dir: Path | str,
    metrics: list[str],
    *,
    write_markdown: bool = True,
) -> dict[str, Path]:
    """Write benchmark results and derived summaries into ``output_dir``."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = _results_dataframe(results, metrics)

    csv_path = output_dir / "benchmark_results.csv"
    json_path = output_dir / "benchmark_results.json"
    summary_path = output_dir / "benchmark_summary.json"
    markdown_path = output_dir / "benchmark_summary.md"

    df.to_csv(csv_path, index=False)
    safe_rows = [_json_safe_dict(row) for row in df.to_dict(orient="records")]
    json_path.write_text(json.dumps({"metrics": metrics, "rows": safe_rows}, indent=2))

    summary = summarize_results(results, metrics)
    summary_path.write_text(json.dumps(summary, indent=2))

    paths = {
        "csv": csv_path,
        "json": json_path,
        "summary_json": summary_path,
    }
    if write_markdown:
        markdown_path.write_text(build_markdown_summary(results, metrics, summary))
        paths["markdown"] = markdown_path

    logger.info("Wrote benchmark report bundle under %s", output_dir)
    return paths


def summarize_results(results: list[dict[str, Any]], metrics: list[str]) -> dict[str, Any]:
    completed = [r for r in results if r.get("status", "ok") == "ok"]
    failed = [r for r in results if r.get("status", "ok") != "ok"]
    ranking_metric = "rmse" if "rmse" in metrics else (metrics[0] if metrics else None)
    ranking = []
    if ranking_metric is not None:
        ranking = [
            {"rank": idx, "model": row.get("model"), ranking_metric: _clean_number(row.get(ranking_metric))}
            for idx, row in enumerate(_sort_rows(completed, ranking_metric), start=1)
        ]

    winners: dict[str, Any] = {}
    for metric in metrics:
        metric_rows = [r for r in completed if _is_finite_number(r.get(metric))]
        if not metric_rows:
            continue
        reverse = metric == "coverage_95"
        best = sorted(metric_rows, key=lambda r: float(r[metric]), reverse=reverse)[0]
        winners[metric] = {"model": best.get("model"), "value": _clean_number(best.get(metric))}

    return {
        "n_models_total": len(results),
        "n_models_completed": len(completed),
        "n_models_failed": len(failed),
        "ranking_metric": ranking_metric,
        "ranking": ranking,
        "metric_winners": winners,
        "failed_models": [
            {"model": row.get("model"), "status": row.get("status"), "error": row.get("error")}
            for row in failed
        ],
        "method_comparison": method_rows([str(r.get("model")) for r in results if r.get("model")]),
    }


def build_markdown_summary(
    results: list[dict[str, Any]],
    metrics: list[str],
    summary: dict[str, Any] | None = None,
) -> str:
    if summary is None:
        summary = summarize_results(results, metrics)
    lines = [
        "# Benchmark Summary",
        "",
        f"Models evaluated: {summary['n_models_total']}  ",
        f"Completed: {summary['n_models_completed']}  ",
        f"Failed/skipped: {summary['n_models_failed']}",
        "",
    ]

    ranking_metric = summary.get("ranking_metric")
    if ranking_metric and summary.get("ranking"):
        lines.extend([
            "## Leaderboard",
            "",
            f"Primary ranking metric: `{ranking_metric}`",
            "",
            "| Rank | Model | Value |",
            "| --- | --- | --- |",
        ])
        for row in summary["ranking"]:
            lines.append(f"| {row['rank']} | {row['model']} | {_fmt(row.get(ranking_metric))} |")
        lines.append("")

    if summary.get("metric_winners"):
        lines.extend(["## Metric Winners", "", "| Metric | Best model | Value |", "| --- | --- | --- |"])
        for metric, row in summary["metric_winners"].items():
            lines.append(f"| {metric} | {row['model']} | {_fmt(row['value'])} |")
        lines.append("")

    lines.extend([
        "## Method Comparison",
        "",
        "| Model | Family | Probabilistic | Physics-informed | Grid requirement | Compute profile |",
        "| --- | --- | --- | --- | --- | --- |",
    ])
    for row in method_rows([str(r.get("model")) for r in results if r.get("model")]):
        lines.append(
            f"| {row['model']} | {row['family']} | {'yes' if row['probabilistic'] else 'no'} | {'yes' if row['physics_informed'] else 'no'} | {row['grid_requirement']} | {row['compute_profile']} |"
        )
    lines.append("")

    if summary.get("failed_models"):
        lines.extend(["## Failures / Skips", ""])
        for row in summary["failed_models"]:
            lines.append(f"- `{row['model']}`: {row.get('status', 'unknown')} ({row.get('error', 'no error message')})")
        lines.append("")

    lines.extend([
        "## Interpretation Checklist",
        "",
        "- Check whether the best RMSE model is also acceptable in runtime and artifact size.",
        "- Compare probabilistic models on both accuracy and uncertainty quality; do not interpret NLL or coverage for non-UQ methods.",
        "- Use the saved prediction files to inspect error concentration over Q2, W2, and |t| instead of relying only on aggregate scores.",
        "- Separate deterministic ranking claims from physics-consistency claims, especially when constraint violation is near zero for all models.",
        "",
    ])
    return "\n".join(lines)


def print_leaderboard(results: list[dict[str, Any]], metrics: list[str]) -> None:
    """Print an ASCII leaderboard ranked by RMSE (lower=better)."""
    if not results:
        print("(no results)")
        return

    ranking_metric = "rmse" if "rmse" in metrics else (metrics[0] if metrics else "fit_time_sec")
    ranked = _sort_rows([r for r in results if r.get("status", "ok") == "ok"], ranking_metric)
    remaining = [r for r in results if r.get("status", "ok") != "ok"]
    sorted_rows = ranked + remaining

    cols = ["model", "status"] + metrics + ["fit_time_sec"]
    widths = {c: max(len(c), 6) for c in cols}
    for row in sorted_rows:
        for col in cols:
            widths[col] = max(widths[col], len(_fmt(row.get(col))))

    sep = "+" + "+".join("-" * (widths[c] + 2) for c in cols) + "+"
    header = "|" + "|".join(f" {c:<{widths[c]}} " for c in cols) + "|"
    print(sep)
    print(header)
    print(sep)
    for row in sorted_rows:
        cells = []
        for col in cols:
            cells.append(f" {_fmt(row.get(col)):<{widths[col]}} ")
        print("|" + "|".join(cells) + "|")
    print(sep)


def _results_dataframe(results: list[dict[str, Any]], metrics: list[str]) -> pd.DataFrame:
    columns = ["model"] + list(metrics) + [c for c in _STANDARD_RESULT_COLUMNS if c != "model"]
    extra = sorted({k for row in results for k in row if k not in columns})
    final_columns = columns + extra
    rows = [{column: row.get(column, None) for column in final_columns} for row in results]
    return pd.DataFrame(rows, columns=final_columns)


def _sort_rows(rows: list[dict[str, Any]], metric: str) -> list[dict[str, Any]]:
    if metric == "coverage_95":
        return sorted(rows, key=lambda row: (not _is_finite_number(row.get(metric)), -float(row.get(metric)) if _is_finite_number(row.get(metric)) else 0.0))
    return sorted(rows, key=lambda row: (not _is_finite_number(row.get(metric)), float(row.get(metric)) if _is_finite_number(row.get(metric)) else float("inf")))


def _json_safe_dict(row: dict[str, Any]) -> dict[str, Any]:
    return {key: _clean_number(value) for key, value in row.items()}


def _clean_number(value: Any) -> Any:
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _fmt(value: Any) -> str:
    if value is None:
        return "–"
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if abs(value) < 1e-3 or abs(value) >= 1e5:
            return f"{value:.3e}"
        return f"{value:.4f}"
    if isinstance(value, bool):
        return "yes" if value else "no"
    return str(value)
