"""Training: prepared splits -> fit on train -> validate -> single test evaluation -> version -> report."""
from __future__ import annotations

import json
import logging
import re
import shutil
from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np
from sklearn.pipeline import Pipeline

from . import evaluate as ev
from .config import Config
from .drift import build_baseline
from .features import build_preprocessor, select_columns
from .models import build_model
from .prepare import ensure_prepared, load_split
from .versioning import environment, git_commit, new_run_id

log = logging.getLogger(__name__)

README_START, README_END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"
PERCENT_METRICS = {"accuracy", "balanced_accuracy"}


def _fmt(name: str, value: float) -> str:
    return f"{value * 100:.2f}%" if name in PERCENT_METRICS else f"{value:.4f}"


def render_results(res: dict) -> str:
    primary = res["primary_metric"]
    n = res["rows"]
    lines = [
        f"**Test {primary} = {_fmt(primary, res['test_metrics'][primary])}**",
        "",
        f"`{res['model']}` ({res['task']}) | run `{res['run_id']}` | commit `{res['git_commit'] or 'n/a'}` | "
        f"rows train/val/test: {n['train']:,}/{n['val']:,}/{n['test']:,} | features: {len(res['features'])}",
        "",
        "| Metric | Validation | Test | Baseline (test) |",
        "|---|---|---|---|",
    ]
    for name, value in res["test_metrics"].items():
        val = res["val_metrics"].get(name)
        base = res["baseline_test_metrics"].get(name)
        lines.append(
            f"| {name} | {_fmt(name, val) if val is not None else '-'} | {_fmt(name, value)} |"
            f" {_fmt(name, base) if base is not None else '-'} |"
        )
    lines.append("")
    lines += [f"![{f}](reports/figures/{f})" for f in res["figures"]]
    return "\n".join(lines).strip() + "\n"


def update_readme(markdown: str, readme: str | Path = "README.md") -> bool:
    path = Path(readme)
    if not path.exists():
        return False
    text = path.read_text(encoding="utf-8")
    pattern = re.compile(re.escape(README_START) + r".*?" + re.escape(README_END), re.DOTALL)
    if not pattern.search(text):
        log.warning("README markers not found; skipping README update.")
        return False
    block = f"{README_START}\n{markdown}{README_END}"
    path.write_text(pattern.sub(lambda _: block, text), encoding="utf-8")
    return True


def run_training(cfg: Config, readme: str | Path = "README.md") -> dict:
    prep = ensure_prepared(cfg)
    task = prep["task"]
    log.info("Task: %s | train/val/test rows: %s", task, prep["rows"])

    # Only train and validation data are loaded while the model is being built.
    X_train, y_train = load_split(cfg, "train")
    X_val, y_val = load_split(cfg, "val")
    numeric, categorical = select_columns(X_train, cfg.data.max_categorical_cardinality)
    features = numeric + categorical
    X_train, X_val = X_train[features], X_val[features]

    pipeline = Pipeline(
        [
            ("preprocess", build_preprocessor(numeric, categorical)),
            ("model", build_model(task, cfg.model.name, cfg.model.params, cfg.data.random_state)),
        ]
    )
    pipeline.fit(X_train, y_train)
    val_metrics, _ = ev.evaluate_pipeline(task, pipeline, X_val, y_val)

    # The test split is first read here, after the model is fixed, and passed through the model
    # exactly once; the plot below reuses these predictions.
    X_test, y_test = load_split(cfg, "test")
    test_metrics, test_pred = ev.evaluate_pipeline(task, pipeline, X_test[features], y_test)

    baseline = ev.baseline_model(task).fit(np.zeros((len(y_train), 1)), y_train)
    baseline_val = ev.compute_metrics(task, y_val, baseline.predict(np.zeros((len(y_val), 1))))
    baseline_metrics = ev.compute_metrics(task, y_test, baseline.predict(np.zeros((len(y_test), 1))))

    run_id = new_run_id()
    reports = Path(cfg.reports_dir)
    figures = ev.save_figures(
        task, pipeline, y_test, test_pred, (X_val, y_val), reports / "figures", cfg.data.random_state
    )

    results = {
        "run_id": run_id,
        "git_commit": git_commit(),
        "task": task,
        "model": cfg.model.name,
        "model_params": cfg.model.params,
        "primary_metric": ev.PRIMARY_METRIC[task],
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "baseline_val_metrics": baseline_val,
        "baseline_test_metrics": baseline_metrics,
        "rows": prep["rows"],
        "features": features,
        "numeric_features": numeric,
        "categorical_features": categorical,
        "classes": [str(c) for c in pipeline.classes_] if task == "classification" else None,
        "data": {
            "path": prep["raw_path"],
            "sha256": prep["raw_sha256"],
            "raw_rows": prep["raw_rows"],
            "clean_rows": prep["clean_rows"],
            "missing_before_imputation": prep["missing_before_imputation"],
            "imputation_values_from_train": prep["imputation_values_from_train"],
        },
        "config": asdict(cfg),
        "environment": environment(),
        "figures": figures,
    }

    # Versioned artifact per run + a stable "latest" copy for inference
    run_dir = Path(cfg.output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, run_dir / "model.joblib")
    (run_dir / "metadata.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    latest = Path(cfg.output_dir) / "latest"
    if latest.exists():
        shutil.rmtree(latest)
    shutil.copytree(run_dir, latest)

    reports.mkdir(parents=True, exist_ok=True)
    (reports / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (reports / "baseline_stats.json").write_text(
        json.dumps({**build_baseline(X_train, numeric, categorical), "cast": cfg.data.categorical_columns}),
        encoding="utf-8",
    )
    markdown = render_results(results)
    (reports / "results.md").write_text(markdown, encoding="utf-8")
    if cfg.update_readme and update_readme(markdown, readme):
        log.info("README results block updated.")

    log.info("Validation metrics: %s", val_metrics)
    log.info("Test metrics: %s", test_metrics)
    return results
