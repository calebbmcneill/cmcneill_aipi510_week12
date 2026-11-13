"""Batch inference and drift checks against a trained model."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from .data import load_dataset, prepare_features
from .drift import compute_drift


def load_model(model_dir: str | Path = "models/latest"):
    model_dir = Path(model_dir)
    model_path = model_dir / "model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"No model at '{model_path}'. Run `mlpipeline train` first.")
    meta = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
    return joblib.load(model_path), meta


def predict_file(
    input_path: str | Path, output_path: str | Path, model_dir: str | Path = "models/latest"
) -> pd.DataFrame:
    pipeline, meta = load_model(model_dir)
    df = load_dataset(input_path)
    missing = [c for c in meta["features"] if c not in df.columns]
    if missing:
        raise ValueError(f"Input is missing required feature columns: {missing}")
    X = prepare_features(df, meta["config"]["data"]["categorical_columns"])[meta["features"]]
    out = df.copy()
    out["prediction"] = pipeline.predict(X)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    return out


def drift_file(
    input_path: str | Path,
    baseline_path: str | Path = "reports/baseline_stats.json",
    output_path: str | Path | None = "reports/drift_report.csv",
) -> pd.DataFrame:
    baseline_path = Path(baseline_path)
    if not baseline_path.exists():
        raise FileNotFoundError(f"No baseline at '{baseline_path}'. Run `mlpipeline train` first.")
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    report = compute_drift(baseline, prepare_features(load_dataset(input_path), baseline.get("cast", [])))
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        report.to_csv(output_path, index=False)
    return report
