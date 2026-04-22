"""Benchmark subcommand — Mode 1."""

from __future__ import annotations

import logging

from diffsurrogate.config import Config
from diffsurrogate.evaluation.benchmark import run_benchmark

logger = logging.getLogger(__name__)


def run(cfg: Config) -> int:
    """Execute the benchmark pipeline. Returns 0 on success."""
    run_benchmark(cfg)
    return 0
