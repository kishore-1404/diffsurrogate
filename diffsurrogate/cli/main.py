"""``diffsurrogate`` command-line entrypoint.

Three subcommands:

  diffsurrogate benchmark --config config.toml
  diffsurrogate train     --config config.toml
  diffsurrogate predict   --config config.toml [--models name1,name2]

Each subcommand loads a TOML config and dispatches to its ``run()`` function
in the matching ``*_cmd`` module.
"""

from __future__ import annotations

import argparse
import logging
import sys

from diffsurrogate.config import load_config

logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="diffsurrogate",
        description="Surrogate modeling for diffractive scattering observables.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # benchmark
    p_bench = sub.add_parser(
        "benchmark",
        help="Compare all enabled surrogates head-to-head against a held-out test set.",
    )
    p_bench.add_argument("--config", "-c", required=True, type=str,
                         help="Path to config.toml")

    # train
    p_train = sub.add_parser(
        "train",
        help="Fit all enabled models on the full dataset and save production artifacts.",
    )
    p_train.add_argument("--config", "-c", required=True, type=str,
                         help="Path to config.toml")

    # predict
    p_pred = sub.add_parser(
        "predict",
        help="Load production models and run inference on new kinematic points.",
    )
    p_pred.add_argument("--config", "-c", required=True, type=str,
                        help="Path to config.toml")
    p_pred.add_argument(
        "--models", "-m", default=None, type=str,
        help="Comma-separated model names to use. Defaults to all enabled.",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    """Return process exit code."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        cfg = load_config(args.config)
    except Exception as e:  # noqa: BLE001
        print(f"ERROR: failed to load config {args.config}: {e}", file=sys.stderr)
        return 2

    if args.command == "benchmark":
        from diffsurrogate.cli.benchmark_cmd import run as run_cmd
        return run_cmd(cfg)
    if args.command == "train":
        from diffsurrogate.cli.train_cmd import run as run_cmd
        return run_cmd(cfg)
    if args.command == "predict":
        from diffsurrogate.cli.predict_cmd import run as run_cmd
        models = None
        if args.models:
            models = [m.strip() for m in args.models.split(",") if m.strip()]
        return run_cmd(cfg, model_names=models)

    parser.print_help(sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
