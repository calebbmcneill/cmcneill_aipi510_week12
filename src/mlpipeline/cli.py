"""Command line interface: `mlpipeline train | predict | drift`."""
from __future__ import annotations

import argparse
import logging

from .config import load_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mlpipeline", description=__doc__)
    sub = parser.add_subparsers(dest="command")

    prep = sub.add_parser("prepare", help="clean raw data and write train/val/test splits")
    prep.add_argument("--config", default="config.yaml")

    train = sub.add_parser("train", help="prepare (if needed), train, evaluate, version and report")
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
    if args.command == "prepare":
        from .prepare import prepare

        report = prepare(load_config(args.config))
        print(f"Rows (train/val/test): {report['rows']} | imputed from train: "
              f"{report['imputation_values_from_train'] or 'nothing'}")
    elif args.command in (None, "train"):
        from .train import run_training

        results = run_training(load_config(getattr(args, "config", "config.yaml")))
        primary = results["primary_metric"]
        print(f"\nValidation {primary}: {results['val_metrics'][primary]:.4f}")
        print(f"Test {primary}: {results['test_metrics'][primary]:.4f}")
    elif args.command == "predict":
        from .predict import predict_file

        out = predict_file(args.input, args.output, args.model_dir)
        print(f"Wrote {len(out):,} predictions to {args.output}")
    elif args.command == "drift":
        from .predict import drift_file

        report = drift_file(args.input, args.baseline, args.output)
        print(report.to_string(index=False))
    return 0
