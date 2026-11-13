"""Metrics, baselines and diagnostic plots."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.dummy import DummyClassifier, DummyRegressor  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

PRIMARY_METRIC = {"classification": "accuracy", "regression": "r2"}


def baseline_model(task: str):
    return DummyClassifier(strategy="most_frequent") if task == "classification" else DummyRegressor()


def compute_metrics(task: str, y_true, y_pred, y_proba=None, classes=None) -> dict:
    if task == "regression":
        return {
            "r2": float(r2_score(y_true, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "mae": float(mean_absolute_error(y_true, y_pred)),
        }
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
    }
    if y_proba is not None:
        try:
            if len(classes) == 2:
                metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba[:, 1]))
            else:
                metrics["roc_auc"] = float(
                    roc_auc_score(y_true, y_proba, multi_class="ovr", labels=list(classes))
                )
        except ValueError:  # e.g. a class is absent from the test split
            pass
    return metrics


def evaluate_pipeline(task: str, pipeline, X, y) -> tuple[dict, np.ndarray]:
    """Score `pipeline` on (X, y) with a single pass of the model over X.

    Returns (metrics, predictions) so callers can reuse the predictions instead of re-predicting.
    """
    y_proba = None
    if task == "classification" and hasattr(pipeline, "predict_proba"):
        y_proba = pipeline.predict_proba(X)
        y_pred = pipeline.classes_[y_proba.argmax(axis=1)]  # same labels predict() would return
    else:
        y_pred = pipeline.predict(X)
    classes = getattr(pipeline, "classes_", None)
    return compute_metrics(task, y, y_pred, y_proba, classes), y_pred


def save_figures(task: str, pipeline, test_true, test_pred, val, fig_dir: Path, seed: int = 42) -> list[str]:
    """Diagnostic plot from the test predictions; permutation importance on the validation split."""
    y_test, y_pred = test_true, test_pred
    X_val, y_val = val
    fig_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    fig, ax = plt.subplots(figsize=(6, 5))
    if task == "classification":
        ConfusionMatrixDisplay.from_predictions(
            y_test, y_pred, ax=ax, cmap="Blues", normalize="true", values_format=".2f"
        )
        ax.set_title("Confusion matrix (test, row-normalised)")
        name = "confusion_matrix.png"
    else:
        ax.scatter(y_test, y_pred, s=10, alpha=0.5)
        lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
        ax.plot(lims, lims, "r--", lw=1)
        ax.set(xlabel="Actual", ylabel="Predicted", title="Predicted vs actual (test)")
        name = "predicted_vs_actual.png"
    fig.tight_layout()
    fig.savefig(fig_dir / name, dpi=120)
    plt.close(fig)
    saved.append(name)

    # Permutation importance on original columns (model-agnostic, works through the pipeline)
    sample = X_val if len(X_val) <= 2000 else X_val.sample(2000, random_state=seed)
    result = permutation_importance(
        pipeline, sample, y_val.loc[sample.index], n_repeats=5, random_state=seed, n_jobs=1
    )
    imp = pd.Series(result.importances_mean, index=X_val.columns).sort_values().tail(15)
    fig, ax = plt.subplots(figsize=(7, 0.35 * len(imp) + 1.5))
    imp.plot.barh(ax=ax)
    ax.set_title("Permutation importance (top features, validation)")
    fig.tight_layout()
    fig.savefig(fig_dir / "feature_importance.png", dpi=120)
    plt.close(fig)
    saved.append("feature_importance.png")
    return saved
