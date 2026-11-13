"""Preprocessing: impute, scale numeric, one-hot encode categorical."""
from __future__ import annotations

import logging

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

log = logging.getLogger(__name__)


def select_columns(X: pd.DataFrame, max_cardinality: int) -> tuple[list[str], list[str]]:
    """Return (numeric, categorical) columns, dropping unusable ones with a warning."""
    numeric, categorical = [], []
    for col in X.columns:
        series = X[col]
        if series.nunique(dropna=True) <= 1:
            log.warning("Dropping constant/empty column '%s'.", col)
        elif pd.api.types.is_numeric_dtype(series):
            numeric.append(col)
        elif pd.api.types.is_datetime64_any_dtype(series):
            log.warning("Dropping datetime column '%s' (engineer date parts first).", col)
        elif series.nunique() > max_cardinality:
            log.warning(
                "Dropping high-cardinality column '%s' (%d unique values).", col, series.nunique()
            )
        else:
            categorical.append(col)
    if not numeric and not categorical:
        raise ValueError("No usable feature columns remain after preprocessing.")
    return numeric, categorical


def build_preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    transformers = []
    if numeric:
        transformers.append(
            (
                "num",
                Pipeline(
                    [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
                ),
                numeric,
            )
        )
    if categorical:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical,
            )
        )
    return ColumnTransformer(transformers, remainder="drop")
