**Test r2 = 0.8925** (validation: 0.8828, baseline: -0.0003)

- Task: `regression` | Model: `random_forest` | Run: `20261113T160957Z` | Commit: `c3339a4`
- Data: `data/raw/auto-mpg.data` (398 raw rows, 398 after cleaning, SHA-256 `48b830e11fee`)
- Split (train/val/test): 278 / 60 / 60. The test split is evaluated once, after the model is fit.
- Features used: 7

| Metric | Validation | Test | Baseline (test) |
|---|---|---|---|
| r2 | 0.8828 | 0.8925 | -0.0003 |
| rmse | 2.3984 | 2.6573 | 8.1066 |
| mae | 1.8211 | 1.7306 | 6.7100 |

![predicted_vs_actual.png](reports/figures/predicted_vs_actual.png)
![feature_importance.png](reports/figures/feature_importance.png)
