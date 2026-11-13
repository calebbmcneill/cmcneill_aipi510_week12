"""Model registry. Add a new model by adding one line to the relevant dict."""
from __future__ import annotations

from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge

# name -> (estimator class, default kwargs)
CLASSIFIERS = {
    "logistic_regression": (LogisticRegression, {"max_iter": 1000}),
    "random_forest": (RandomForestClassifier, {"n_estimators": 200, "n_jobs": -1}),
    "gradient_boosting": (HistGradientBoostingClassifier, {}),
}
REGRESSORS = {
    "linear_regression": (LinearRegression, {}),
    "ridge": (Ridge, {}),
    "random_forest": (RandomForestRegressor, {"n_estimators": 200, "n_jobs": -1}),
    "gradient_boosting": (HistGradientBoostingRegressor, {}),
}
# Estimators that accept random_state, so runs are reproducible
SEEDED = {RandomForestClassifier, RandomForestRegressor, HistGradientBoostingClassifier,
          HistGradientBoostingRegressor, LogisticRegression}


def build_model(task: str, name: str, params: dict | None = None, seed: int = 42):
    registry = CLASSIFIERS if task == "classification" else REGRESSORS
    if name not in registry:
        raise ValueError(f"Unknown {task} model '{name}'. Choose from {sorted(registry)}.")
    cls, defaults = registry[name]
    kwargs = {**defaults, **(params or {})}
    if cls in SEEDED:
        kwargs.setdefault("random_state", seed)
    return cls(**kwargs)
