"""Data loading, validation and splitting."""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import Config

log = logging.getLogger(__name__)

# A numeric target with more distinct values than this is treated as regression.
MAX_CLASSES_FOR_AUTO = 20


def load_dataset(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{path}'. Add it under data/raw/ and set data.path in config.yaml."
        )
    readers = {
        ".csv": pd.read_csv,
        ".tsv": lambda p: pd.read_csv(p, sep="\t"),
        ".json": pd.read_json,
        ".parquet": pd.read_parquet,
        ".xlsx": pd.read_excel,
    }
    reader = readers.get(path.suffix.lower())
    if reader is None:
        raise ValueError(f"Unsupported file type '{path.suffix}'. Use one of {sorted(readers)}.")
    return reader(path)


def validate(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Basic data-quality gate: fail loudly on structural problems, warn on soft ones."""
    target = cfg.data.target
    if target not in df.columns:
        raise ValueError(
            f"Target column '{target}' not found. Available columns: {list(df.columns)}"
        )
    missing_drop = [c for c in cfg.data.drop_columns if c not in df.columns]
    if missing_drop:
        log.warning("drop_columns not present in data (ignored): %s", missing_drop)

    n_before = len(df)
    df = df.dropna(subset=[target]).drop_duplicates().reset_index(drop=True)
    if len(df) < n_before:
        log.warning("Removed %d rows with missing target or duplicated.", n_before - len(df))
    if len(df) < 20:
        raise ValueError(f"Only {len(df)} usable rows; need at least 20 to split and evaluate.")

    if cfg.data.sample_rows and len(df) > cfg.data.sample_rows:
        df = df.sample(cfg.data.sample_rows, random_state=cfg.data.random_state).reset_index(
            drop=True
        )
        log.info("Sampled %d rows (data.sample_rows).", cfg.data.sample_rows)
    return df


def infer_task(y: pd.Series, requested: str = "auto") -> str:
    if requested != "auto":
        return requested
    if pd.api.types.is_numeric_dtype(y) and not pd.api.types.is_bool_dtype(y):
        if y.nunique() > MAX_CLASSES_FOR_AUTO:
            return "regression"
    return "classification"


def prepare_features(X: pd.DataFrame) -> pd.DataFrame:
    """Shared by training and inference so both see identically typed columns."""
    X = X.copy()
    for col in X.select_dtypes(include="bool").columns:
        X[col] = X[col].astype(int)
    return X


def split_xy(df: pd.DataFrame, cfg: Config, task: str):
    y = df[cfg.data.target]
    X = prepare_features(df.drop(columns=[cfg.data.target, *cfg.data.drop_columns], errors="ignore"))

    stratify = None
    if task == "classification" and cfg.data.stratify and y.value_counts().min() >= 2:
        stratify = y
    return train_test_split(
        X,
        y,
        test_size=cfg.data.test_size,
        random_state=cfg.data.random_state,
        stratify=stratify,
    )
