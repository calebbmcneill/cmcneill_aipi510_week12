**Test R² = 0.8925** (89.3% of the variance in `mpg` explained) | RMSE 2.66 `mpg` | MAE 1.73 `mpg`

`random_forest` (regression) | run `20261113T164628Z` | commit `6cf8774` | rows train/val/test: 278/60/60 | features: 7

| Metric | Validation | Test | Baseline (test) |
|---|---|---|---|
| r2 | 0.8828 | 0.8925 | -0.0003 |
| rmse | 2.3984 | 2.6573 | 8.1066 |
| mae | 1.8211 | 1.7306 | 6.7100 |

![predicted_vs_actual.png](reports/figures/predicted_vs_actual.png)
![feature_importance.png](reports/figures/feature_importance.png)
