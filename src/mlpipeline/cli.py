"""Command line interface: `mlpipeline train | predict | drift`."""
from __future__ import annotations

import argparse
import logging

from .config import load_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mlpipeline", description=__doc__)
    sub = parser.add_subparsers(dest="command")

    train = sub.add_parser("train", help="train, evaluate, version and report")
    train.add_argument("--config", default="config.yaml")

    pred = sub.add_parser("predict", help="score a file with the latest model")
    pred.add_argument("--input", required=True)
    pred.add_argument("--output", default="reports/predictions.csv")
    pred.add_argument("--model-dir", default="models/latest")

    drift = sub.add_parser("drift", help="compare new data against the training distribution")
    drift.add_argument("--input", required=True)
    drift.add_argument("--baseline", default="reports/baseline_stats.json")
    drift.add_argument("--output", default="reports/drift_report.csv")

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    # Imports are deferred so `--help` stays fast.
    if args.command in (None, "train"):
        from .train import run_training

        results = run_training(load_config(getattr(args, "config", "config.yaml")))
        primary = results["primary_metric"]
        print(f"\nTest {primary}: {results['test_metrics'][primary]:.4f}")
    elif args.command == "predict":
        from .predict import predict_file

        out = predict_file(args.input, args.output, args.model_dir)
        print(f"Wrote {len(out):,} predictions to {args.output}")
    elif args.command == "drift":
        from .predict import drift_file

        report = drift_file(args.input, args.baseline, args.output)
        print(report.to_string(index=False))
    return 0
