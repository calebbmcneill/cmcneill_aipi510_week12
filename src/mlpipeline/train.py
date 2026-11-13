"""End-to-end training: load -> validate -> split -> fit -> evaluate -> version -> report."""
from __future__ import annotations

import json
import logging
import re
import shutil
from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

from . import evaluate as ev
from .config import Config
from .data import infer_task, load_dataset, split_xy, validate
from .drift import build_baseline
from .features import build_preprocessor, select_columns
from .models import build_model
from .versioning import environment, file_sha256, git_commit, new_run_id

log = logging.getLogger(__name__)

README_START, README_END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"
PERCENT_METRICS = {"accuracy", "balanced_accuracy"}


def _fmt(name: str, value: float) -> str:
    return f"{value * 100:.2f}%" if name in PERCENT_METRICS else f"{value:.4f}"


def render_results(res: dict) -> str:
    primary = res["primary_metric"]
    label = primary.replace("_", " ")
    lines = [
        f"**Test {label} = {_fmt(primary, res['test_metrics'][primary])}**"
        f" (baseline: {_fmt(primary, res['baseline_test_metrics'][primary])})",
        "",
        f"- Task: `{res['task']}` | Model: `{res['model']}` | Run: `{res['run_id']}`"
        f" | Commit: `{res['git_commit'] or 'n/a'}`",
        f"- Data: `{res['data']['path']}` ({res['data']['rows']:,} rows, "
        f"SHA-256 `{res['data']['sha256'][:12]}`) | Train/test: {res['n_train']:,}/{res['n_test']:,}",
        f"- Features used: {len(res['features'])}",
        "",
        "| Metric | Test | Baseline (test) |",
        "|---|---|---|",
    ]
    for name, value in res["test_metrics"].items():
        base = res["baseline_test_metrics"].get(name)
        lines.append(f"| {name} | {_fmt(name, value)} | {_fmt(name, base) if base is not None else '-'} |")
    cv = res.get("cv")
    if cv:
        pct = cv["metric"] in PERCENT_METRICS
        std = f"{cv['std'] * 100:.2f}%" if pct else f"{cv['std']:.4f}"
        lines += [
            "",
            f"Cross-validation ({cv['folds']}-fold, train split, {cv['metric']}): "
            f"{_fmt(cv['metric'], cv['mean'])} ± {std}",
        ]
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
    data_path = Path(cfg.data.path)
    df = validate(load_dataset(data_path), cfg)
    task = infer_task(df[cfg.data.target], cfg.task)
    log.info("Task: %s | rows: %d", task, len(df))

    X_train, X_test, y_train, y_test = split_xy(df, cfg, task)
    numeric, categorical = select_columns(X_train, cfg.data.max_categorical_cardinality)
    features = numeric + categorical
    X_train, X_test = X_train[features], X_test[features]

    pipeline = Pipeline(
        [
            ("preprocess", build_preprocessor(numeric, categorical)),
            ("model", build_model(task, cfg.model.name, cfg.model.params, cfg.data.random_state)),
        ]
    )

    cv_result = None
    folds = cfg.cv_folds
    if task == "classification":
        folds = min(folds, int(y_train.value_counts().min()))
    if folds >= 2:
        metric = ev.CV_SCORING[task]
        scores = cross_val_score(clone(pipeline), X_train, y_train, cv=folds, scoring=metric)
        cv_result = {
            "folds": folds,
            "metric": ev.PRIMARY_METRIC[task],
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
        }

    pipeline.fit(X_train, y_train)
    test_metrics = ev.evaluate_pipeline(task, pipeline, X_test, y_test)

    baseline = ev.baseline_model(task).fit(np.zeros((len(y_train), 1)), y_train)
    baseline_pred = baseline.predict(np.zeros((len(y_test), 1)))
    baseline_metrics = ev.compute_metrics(task, y_test, baseline_pred)

    run_id = new_run_id()
    reports = Path(cfg.reports_dir)
    figures = ev.save_figures(task, pipeline, X_test, y_test, reports / "figures", cfg.data.random_state)

    results = {
        "run_id": run_id,
        "git_commit": git_commit(),
        "task": task,
        "model": cfg.model.name,
        "model_params": cfg.model.params,
        "primary_metric": ev.PRIMARY_METRIC[task],
        "test_metrics": test_metrics,
        "baseline_test_metrics": baseline_metrics,
        "cv": cv_result,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "features": features,
        "numeric_features": numeric,
        "categorical_features": categorical,
        "classes": [str(c) for c in pipeline.classes_] if task == "classification" else None,
        "data": {
            "path": data_path.as_posix(),
            "sha256": file_sha256(data_path),
            "rows": len(df),
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
        json.dumps(build_baseline(X_train, numeric, categorical)), encoding="utf-8"
    )
    markdown = render_results(results)
    (reports / "results.md").write_text(markdown, encoding="utf-8")
    if cfg.update_readme and update_readme(markdown, readme):
        log.info("README results block updated.")

    log.info("Test metrics: %s", test_metrics)
    return results
