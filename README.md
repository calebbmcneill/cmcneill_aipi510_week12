# ML Pipeline Template (AIPI 510 – Week 12)

A small, config-driven machine-learning pipeline for tabular data. Drop a dataset into
`data/raw/`, set the target column in [`config.yaml`](config.yaml), run one command, and get a
trained model, evaluation metrics, diagnostic plots, versioned artifacts and a drift baseline.
The **Results** section below is rewritten automatically after every training run.

> **Using this as a template:** on GitHub go to *Settings → General → Template repository*, or
> click *Use this template* to start a fresh repo. Then follow *Adapt to your dataset*.

## Results

<!-- RESULTS:START -->
_No model trained yet. Run `mlpipeline train` and this block fills in automatically
with test metrics and plots._
<!-- RESULTS:END -->

## Dataset

_TODO: name, source/URL, license, what one row represents, the target, and why this problem
matters. Full dataset card and data dictionary: [data/README.md](data/README.md)._

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"           # use ".[dev,parquet]" if your data is a .parquet file

mlpipeline train                  # or: python -m mlpipeline train
pytest                            # code + data + model checks
```

## Adapt to your dataset

1. Commit the file to `data/raw/` (keep it immutable; add new versions as new files).
2. Edit [`config.yaml`](config.yaml): `data.path`, `data.target`, and `data.drop_columns` for IDs,
   free text or leakage columns. `task: auto` infers classification vs regression; override if needed.
3. Pick a model (`model.name`) – start simple (`logistic_regression` / `ridge`), then try
   `random_forest` or `gradient_boosting`. Hyperparameters go in `model.params`.
4. `mlpipeline train`, then read the Results block and `reports/figures/`.
5. Fill in the TODOs in this README and in [data/README.md](data/README.md).

## Pipeline

```
data/raw ──► load & validate ──► split (train/test) ──► preprocess + model (one sklearn Pipeline)
                                                              │
        ┌─────────────────────────────────────────────────────┤
        ▼                          ▼                           ▼
 CV on train only        test metrics vs. baseline     versioned artifacts
                         + confusion/residual plot     models/<run_id>/ + models/latest/
                         + permutation importance      reports/metrics.json, baseline_stats.json
```

| Stage | Where | Notes |
|---|---|---|
| Ingest + validate | [`data.py`](src/mlpipeline/data.py) | Fails if the target is missing; drops duplicates and rows with no label; clear error if the file is absent |
| Features | [`features.py`](src/mlpipeline/features.py) | Median impute + scale numerics; mode impute + one-hot categoricals; drops constant, datetime and high-cardinality columns (with warnings) |
| Models | [`models.py`](src/mlpipeline/models.py) | Small registry; add a model with one line |
| Evaluation | [`evaluate.py`](src/mlpipeline/evaluate.py) | Test metrics next to a majority-class/mean baseline, k-fold CV on the training split, plots |
| Versioning | [`versioning.py`](src/mlpipeline/versioning.py) | Each run stores git commit, data SHA-256, full config, library versions and metrics beside the model |
| Monitoring | [`drift.py`](src/mlpipeline/drift.py) | Population Stability Index per feature (<0.1 stable, <0.25 moderate, else significant) |
| CI | [`.github/workflows/ci.yml`](.github/workflows/ci.yml) | Ruff + pytest on every push |
| Packaging | [`Dockerfile`](Dockerfile) | `docker build -t mlpipeline . && docker run mlpipeline` |

The test split is touched once, for final evaluation. Cross-validation uses only the training
split, so the reported test number is an unbiased estimate.

## Using the trained model

```bash
mlpipeline predict --input <new-data-file> --output reports/predictions.csv
mlpipeline drift   --input <new-data-file>     # PSI vs. training distribution
```

## Project layout

```
config.yaml            all dataset/model settings
data/raw/              immutable input data (+ data/README.md dataset card)
src/mlpipeline/        pipeline code (cli, config, data, features, models, evaluate, drift, ...)
tests/                 pytest suite running on synthetic data
reports/               metrics.json, results.md, figures/, drift baseline  (committed)
models/                versioned model artifacts (git-ignored)
```

## Notes and limitations

_TODO: discuss class imbalance, leakage risks, what the model gets wrong, and what you would try
next with more time._