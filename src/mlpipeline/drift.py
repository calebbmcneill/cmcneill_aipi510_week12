"""Data-drift monitoring with the Population Stability Index (PSI).

A baseline of each feature's training distribution is saved at train time; `mlpipeline drift`
compares a new batch of data against it. No labels are required.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

N_BINS = 10
TOP_CATEGORIES = 20
EPS = 1e-4
# Common industry thresholds
STABLE, MODERATE = 0.1, 0.25


def _numeric_props(values: pd.Series, edges: np.ndarray) -> list[float]:
    n = len(values)
    valid = values.dropna().to_numpy(dtype=float)
    counts = np.bincount(np.searchsorted(edges, valid), minlength=len(edges) + 1)
    return [*(counts / n).tolist(), (n - len(valid)) / n]  # last bin = missing


def _categorical_props(values: pd.Series, categories: list[str]) -> list[float]:
    labels = values.astype("object").where(values.notna(), "__missing__").astype(str)
    labels = labels.where(labels.isin(categories + ["__missing__"]), "__other__")
    freq = labels.value_counts(normalize=True)
    return [float(freq.get(c, 0.0)) for c in [*categories, "__other__", "__missing__"]]


def build_baseline(X: pd.DataFrame, numeric: list[str], categorical: list[str]) -> dict:
    baseline: dict = {"numeric": {}, "categorical": {}}
    for col in numeric:
        vals = X[col].dropna().to_numpy(dtype=float)
        edges = np.unique(np.quantile(vals, np.linspace(0, 1, N_BINS + 1)[1:-1]))
        baseline["numeric"][col] = {
            "edges": edges.tolist(),
            "props": _numeric_props(X[col], edges),
        }
    for col in categorical:
        cats = X[col].dropna().astype(str).value_counts().head(TOP_CATEGORIES).index.tolist()
        baseline["categorical"][col] = {"categories": cats, "props": _categorical_props(X[col], cats)}
    return baseline


def psi(expected, actual) -> float:
    e = np.clip(np.asarray(expected, dtype=float), EPS, None)
    a = np.clip(np.asarray(actual, dtype=float), EPS, None)
    return float(np.sum((a - e) * np.log(a / e)))


def status(value: float) -> str:
    if value < STABLE:
        return "stable"
    return "moderate" if value < MODERATE else "significant"


def compute_drift(baseline: dict, df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col, spec in baseline["numeric"].items():
        if col in df:
            actual = _numeric_props(pd.to_numeric(df[col], errors="coerce"), np.array(spec["edges"]))
            rows.append((col, "numeric", psi(spec["props"], actual)))
    for col, spec in baseline["categorical"].items():
        if col in df:
            actual = _categorical_props(df[col], spec["categories"])
            rows.append((col, "categorical", psi(spec["props"], actual)))
    report = pd.DataFrame(rows, columns=["feature", "type", "psi"])
    report["status"] = report["psi"].map(status)
    return report.sort_values("psi", ascending=False).reset_index(drop=True)
