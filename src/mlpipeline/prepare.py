"""Raw -> processed: clean, split into train/val/test, then impute using training data only.

Raw files are never modified. The splits are written to `data.processed_dir`. Imputation values
are learned from the training split alone and then applied to validation and test, so no
information from held-out rows leaks into the model.
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import Config
from .data import infer_task, load_dataset, prepare_features, validate
from .versioning import file_sha256

log = logging.getLogger(__name__)

SPLITS = ("train", "val", "test")
REPORT_NAME = "prepare_report.json"


def split_path(cfg: Config, name: str) -> Path:
    return Path(cfg.data.processed_dir) / f"{name}.csv"


def fingerprint(cfg: Config) -> str:
    """Changes whenever the raw file or any data setting changes."""
    payload = json.dumps(asdict(cfg.data), sort_keys=True, default=str)
    return hashlib.sha256((file_sha256(cfg.data.path) + payload).encode()).hexdigest()


def _split(df: pd.DataFrame, cfg: Config, task: str):
    d = cfg.data
    y = df[d.target]

    def attempt(frame, test_size, labels):
        stratify = labels if task == "classification" and d.stratify else None
        try:
            return train_test_split(frame, test_size=test_size, random_state=d.random_state, stratify=stratify)
        except ValueError:  # a class is too rare to stratify
            log.warning("Could not stratify the split; falling back to a random split.")
            return train_test_split(frame, test_size=test_size, random_state=d.random_state)

    train, rest = attempt(df, d.val_size + d.test_size, y)
    val, test = attempt(rest, d.test_size / (d.val_size + d.test_size), rest[d.target])
    return train, val, test


def prepare(cfg: Config) -> dict:
    d = cfg.data
    raw = load_dataset(d.path, d.read_options)
    df = validate(raw, cfg)
    df = prepare_features(df, d.categorical_columns)
    task = infer_task(df[d.target], cfg.task)

    missing_before = {c: int(n) for c, n in df.isna().sum().items() if n}
    train, val, test = _split(df, cfg, task)

    # Learn fill values from the training split only
    features = [c for c in df.columns if c != d.target]
    imputation = {}
    for col in features:
        if pd.api.types.is_numeric_dtype(train[col]):
            value = train[col].median()
        else:
            modes = train[col].mode(dropna=True)
            value = modes.iloc[0] if len(modes) else None
        if pd.notna(value) and df[col].isna().any():
            imputation[col] = value.item() if hasattr(value, "item") else value

    out_dir = Path(d.processed_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = {}
    for name, frame in zip(SPLITS, (train, val, test)):
        frame = frame.fillna(imputation).reset_index(drop=True)
        frames[name] = frame
        frame.to_csv(split_path(cfg, name), index=False)

    report = {
        "fingerprint": fingerprint(cfg),
        "raw_path": Path(d.path).as_posix(),
        "raw_sha256": file_sha256(d.path),
        "task": task,
        "raw_rows": len(raw),
        "clean_rows": len(df),
        "rows": {name: len(frame) for name, frame in frames.items()},
        "missing_before_imputation": missing_before,
        "missing_after_imputation": {n: int(f.isna().sum().sum()) for n, f in frames.items()},
        "imputation_values_from_train": imputation,
    }
    (out_dir / REPORT_NAME).write_text(json.dumps(report, indent=2), encoding="utf-8")
    log.info(
        "Prepared %d clean rows -> train/val/test = %d/%d/%d | imputed: %s",
        len(df), *report["rows"].values(), imputation or "nothing",
    )
    return report


def ensure_prepared(cfg: Config) -> dict:
    """Re-run preparation if the processed splits are missing or stale."""
    report_path = Path(cfg.data.processed_dir) / REPORT_NAME
    if report_path.exists() and all(split_path(cfg, n).exists() for n in SPLITS):
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("fingerprint") == fingerprint(cfg):
            return report
        log.info("Processed data is stale; re-preparing.")
    return prepare(cfg)


def load_split(cfg: Config, name: str) -> tuple[pd.DataFrame, pd.Series]:
    if name not in SPLITS:
        raise ValueError(f"Unknown split '{name}'")
    df = pd.read_csv(split_path(cfg, name))
    y = df[cfg.data.target]
    X = prepare_features(df.drop(columns=[cfg.data.target]), cfg.data.categorical_columns)
    return X, y
