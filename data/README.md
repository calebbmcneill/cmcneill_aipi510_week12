# Data

Place the dataset in `data/raw/` and point `data.path` in [`config.yaml`](../config.yaml) at it.
Raw data is treated as immutable: never edit it in place, add a new file instead. Every training
run records the file's SHA-256 so a model can always be traced to the exact data it saw.

## Dataset card (fill in)

| Field | Value |
|---|---|
| Name | TODO |
| Source / URL | TODO |
| License | TODO |
| Rows / columns | TODO |
| Target column | TODO |
| Task | TODO (classification or regression) |
| Known issues | TODO (missing values, imbalance, leakage risks) |

## Data dictionary (fill in)

| Column | Type | Description |
|---|---|---|
| TODO | | |
