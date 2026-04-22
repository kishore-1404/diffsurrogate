"""Metrics, benchmark orchestration, plots, reporting."""

from diffsurrogate.evaluation.benchmark import run_benchmark
from diffsurrogate.evaluation.metrics import (
    compute_metrics,
    constraint_violation_rate,
    coverage_95,
    mae,
    nll,
    rmse,
)
from diffsurrogate.evaluation.report import print_leaderboard, write_report
from diffsurrogate.evaluation.visualize import (
    plot_residuals,
    plot_t_spectrum,
    plot_uq_bands,
)

__all__ = [
    "rmse",
    "mae",
    "nll",
    "coverage_95",
    "constraint_violation_rate",
    "compute_metrics",
    "run_benchmark",
    "write_report",
    "print_leaderboard",
    "plot_residuals",
    "plot_uq_bands",
    "plot_t_spectrum",
]
