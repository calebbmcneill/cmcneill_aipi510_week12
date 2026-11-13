"""Data loading, cleaning and feature typing."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config

log = logging.getLogger(__name__)

# A numeric target with more distinct values than this is treated as regression.
MAX_CLASSES_FOR_AUTO = 20


def load_dataset(path: str | Path, read_options: dict | None = None) -> pd.DataFrame:
    """Read a data file. `read_options` is forwarded to the pandas reader (sep, names, na_values...)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{path}'. Add it under data/raw/ and set data.path in config.yaml."
        )
    opts = dict(read_options or {})
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path, **opts)
    if suffix == ".json":
        return pd.read_json(path, **opts)
    if suffix in (".xlsx", ".xls"):
        return pd.read_excel(path, **opts)
    if suffix == ".tsv":
        opts.setdefault("sep", "\t")
    return pd.read_csv(path, **opts)  # .csv and any other delimited text extension


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
    df = df.drop_duplicates().dropna(subset=[target]).reset_index(drop=True)
    if len(df) < n_before:
        log.warning("Removed %d duplicated rows or rows with a missing target.", n_before - len(df))
    if len(df) < 20:
        raise ValueError(f"Only {len(df)} usable rows; need at least 20 to split and evaluate.")

    if cfg.data.sample_rows and len(df) > cfg.data.sample_rows:
        df = df.sample(cfg.data.sample_rows, random_state=cfg.data.random_state).reset_index(
            drop=True
        )
        log.info("Sampled %d rows (data.sample_rows).", cfg.data.sample_rows)
    return df.drop(columns=cfg.data.drop_columns, errors="ignore")


def infer_task(y: pd.Series, requested: str = "auto") -> str:
    if requested != "auto":
        return requested
    if pd.api.types.is_numeric_dtype(y) and not pd.api.types.is_bool_dtype(y):
        if y.nunique() > MAX_CLASSES_FOR_AUTO:
            return "regression"
    return "classification"


def prepare_features(X: pd.DataFrame, categorical_columns: list[str] | None = None) -> pd.DataFrame:
    """Shared by preparation, training and inference so all see identically typed columns."""
    X = X.copy()
    for col in X.select_dtypes(include="bool").columns:
        X[col] = X[col].astype(int)
    for col in categorical_columns or []:
        if col not in X.columns:
            continue
        s = X[col]
        if pd.api.types.is_numeric_dtype(s):
            labels = s.map(lambda v: f"{v:g}" if pd.notna(v) else np.nan)  # 1 and 1.0 -> "1"
        else:
            labels = s.astype(str).where(s.notna(), np.nan)
        X[col] = labels.astype(object)
    return X
