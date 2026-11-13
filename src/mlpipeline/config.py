"""Typed, validated pipeline configuration loaded from YAML."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

TASKS = ("auto", "classification", "regression")


@dataclass
class DataConfig:
    path: str = "data/raw/YOUR_FILE.csv"
    target: str = "target"
    drop_columns: list[str] = field(default_factory=list)
    test_size: float = 0.2
    random_state: int = 42
    stratify: bool = True
    max_categorical_cardinality: int = 50
    sample_rows: int | None = None


@dataclass
class ModelConfig:
    name: str = "random_forest"
    params: dict = field(default_factory=dict)


@dataclass
class Config:
    task: str = "auto"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    cv_folds: int = 5
    output_dir: str = "models"
    reports_dir: str = "reports"
    update_readme: bool = True

    def __post_init__(self) -> None:
        if self.task not in TASKS:
            raise ValueError(f"task must be one of {TASKS}, got {self.task!r}")
        if not 0 < self.data.test_size < 1:
            raise ValueError("data.test_size must be between 0 and 1")


def _build(cls, raw: dict | None, section: str):
    try:
        return cls(**(raw or {}))
    except TypeError as exc:  # unknown keys are almost always typos
        raise ValueError(f"Invalid key in config section '{section}': {exc}") from exc


def load_config(path: str | Path = "config.yaml") -> Config:
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    raw = dict(raw)
    data = _build(DataConfig, raw.pop("data", None), "data")
    model = _build(ModelConfig, raw.pop("model", None), "model")
    return _build(Config, {**raw, "data": data, "model": model}, "root")
