import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression

from mlpipeline.config import Config, DataConfig, ModelConfig, load_config
from mlpipeline.data import infer_task, load_dataset, validate
from mlpipeline.drift import build_baseline, compute_drift
from mlpipeline.predict import drift_file, predict_file
from mlpipeline.prepare import ensure_prepared, prepare, split_path
from mlpipeline.train import run_training, update_readme


def _messy_frame(task: str, n: int = 400) -> pd.DataFrame:
    """Mixed dtypes, missing values, a bool, an ID-like text column: what real data looks like."""
    rng = np.random.default_rng(0)
    if task == "classification":
        X, y = make_classification(n_samples=n, n_features=6, n_informative=4, random_state=0)
    else:
        X, y = make_regression(n_samples=n, n_features=6, n_informative=4, noise=5, random_state=0)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(6)])
    df["color"] = rng.choice(["red", "green", "blue"], n)
    df["flag"] = rng.random(n) > 0.5
    df["row_id"] = [f"id-{i}" for i in range(n)]
    df.loc[rng.choice(n, 20, replace=False), "f0"] = np.nan
    df["target"] = y
    return df


def _cfg(tmp_path, csv, task, model):
    return Config(
        task=task,
        data=DataConfig(path=str(csv), target="target", processed_dir=str(tmp_path / "processed")),
        model=ModelConfig(name=model),
        output_dir=str(tmp_path / "models"),
        reports_dir=str(tmp_path / "reports"),
        update_readme=False,
    )


def test_prepare_splits_70_15_15_without_overlap_and_imputes_from_train(tmp_path):
    csv = tmp_path / "data.csv"
    raw = _messy_frame("regression")
    raw.to_csv(csv, index=False)
    cfg = _cfg(tmp_path, csv, "regression", "ridge")

    report = prepare(cfg)

    parts = {n: pd.read_csv(split_path(cfg, n)) for n in ("train", "val", "test")}
    assert {n: len(p) for n, p in parts.items()} == {"train": 280, "val": 60, "test": 60}
    ids = [set(p["row_id"]) for p in parts.values()]
    assert not (ids[0] & ids[1]) and not (ids[0] & ids[2]) and not (ids[1] & ids[2])
    assert all(p.isna().sum().sum() == 0 for p in parts.values())
    # the fill value is the median of the *training* rows' observed values, not of all rows
    train_raw = raw.set_index("row_id").loc[parts["train"]["row_id"], "f0"]
    assert report["imputation_values_from_train"]["f0"] == pytest.approx(train_raw.median())
    assert report["imputation_values_from_train"]["f0"] != pytest.approx(raw["f0"].median())


def test_test_split_never_influences_the_model(tmp_path):
    csv = tmp_path / "data.csv"
    _messy_frame("regression").to_csv(csv, index=False)
    cfg = _cfg(tmp_path, csv, "regression", "ridge")

    first = run_training(cfg)
    test_path = split_path(cfg, "test")
    test = pd.read_csv(test_path)
    test["target"] = np.random.default_rng(1).permutation(test["target"].to_numpy())
    test.to_csv(test_path, index=False)  # same fingerprint, so splits are not regenerated
    second = run_training(cfg)

    assert first["val_metrics"] == second["val_metrics"]  # model identical
    assert first["test_metrics"] != second["test_metrics"]  # only the test score moved


def test_stale_processed_data_is_regenerated(tmp_path):
    csv = tmp_path / "data.csv"
    _messy_frame("regression", n=400).to_csv(csv, index=False)
    cfg = _cfg(tmp_path, csv, "regression", "ridge")
    assert ensure_prepared(cfg)["clean_rows"] == 400
    _messy_frame("regression", n=300).to_csv(csv, index=False)
    assert ensure_prepared(cfg)["clean_rows"] == 300


def test_load_dataset_applies_read_options(tmp_path):
    path = tmp_path / "raw.data"
    path.write_text('1.0  2  ?  "a b"\n3.0  4  5.5  "c d"\n', encoding="utf-8")
    df = load_dataset(
        path, {"sep": r"\s+", "header": None, "names": ["x", "y", "z", "name"], "na_values": ["?"]}
    )
    assert list(df.columns) == ["x", "y", "z", "name"]
    assert df["z"].isna().tolist() == [True, False] and df["name"][0] == "a b"


@pytest.mark.parametrize(
    "task,model", [("classification", "logistic_regression"), ("regression", "ridge")]
)
def test_train_beats_baseline_and_writes_artifacts(tmp_path, task, model):
    csv = tmp_path / "data.csv"
    _messy_frame(task).to_csv(csv, index=False)

    res = run_training(_cfg(tmp_path, csv, task, model))

    primary = res["primary_metric"]
    assert res["test_metrics"][primary] > res["baseline_test_metrics"][primary]
    assert "row_id" not in res["features"]  # high-cardinality text dropped
    for path in ("models/latest/model.joblib", "models/latest/metadata.json",
                 "reports/metrics.json", "reports/baseline_stats.json", "reports/results.md"):
        assert (tmp_path / path).exists(), path
    assert (tmp_path / "processed" / "test.csv").exists()
    assert len(res["data"]["sha256"]) == 64


def test_predict_and_drift_roundtrip(tmp_path):
    csv = tmp_path / "data.csv"
    df = _messy_frame("classification")
    df.to_csv(csv, index=False)
    run_training(_cfg(tmp_path, csv, "classification", "random_forest"))

    new = df.drop(columns="target").head(25)
    new_csv = tmp_path / "new.csv"
    new.to_csv(new_csv, index=False)
    out = predict_file(new_csv, tmp_path / "preds.csv", tmp_path / "models" / "latest")
    assert len(out) == 25 and "prediction" in out

    bad = tmp_path / "bad.csv"
    new.drop(columns="f1").to_csv(bad, index=False)
    with pytest.raises(ValueError, match="f1"):
        predict_file(bad, tmp_path / "x.csv", tmp_path / "models" / "latest")

    shifted = new.assign(f2=new["f2"] + 5)
    shifted_csv = tmp_path / "shifted.csv"
    shifted.to_csv(shifted_csv, index=False)
    report = drift_file(shifted_csv, tmp_path / "reports" / "baseline_stats.json", None)
    assert report.iloc[0]["feature"] == "f2" and report.iloc[0]["status"] == "significant"


def test_drift_is_stable_on_same_distribution():
    df = _messy_frame("classification", n=2000)
    baseline = build_baseline(df, ["f0", "f1"], ["color"])
    report = compute_drift(baseline, df.sample(1000, random_state=1))
    assert (report["status"] == "stable").all()


def test_validate_rejects_missing_target(tmp_path):
    cfg = Config(data=DataConfig(target="nope"))
    with pytest.raises(ValueError, match="nope"):
        validate(_messy_frame("classification"), cfg)


def test_infer_task():
    assert infer_task(pd.Series([0, 1] * 50)) == "classification"
    assert infer_task(pd.Series(np.linspace(0, 1, 100))) == "regression"
    assert infer_task(pd.Series(["a", "b"] * 50)) == "classification"
    assert infer_task(pd.Series(np.linspace(0, 1, 100)), "classification") == "classification"


def test_config_rejects_unknown_keys(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("data:\n  targt: y\n", encoding="utf-8")
    with pytest.raises(ValueError, match="data"):
        load_config(path)


def test_missing_dataset_has_helpful_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="data/raw"):
        load_dataset(tmp_path / "missing.csv")


def test_update_readme_replaces_only_marked_block(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text("before\n<!-- RESULTS:START -->\nold\n<!-- RESULTS:END -->\nafter\n", encoding="utf-8")
    assert update_readme("NEW\n", readme)
    text = readme.read_text(encoding="utf-8")
    assert "NEW" in text and "old" not in text and text.startswith("before") and text.endswith("after\n")
