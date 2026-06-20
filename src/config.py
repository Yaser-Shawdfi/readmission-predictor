"""
Enterprise Configuration — YAML + env var overrides.
"""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

CONFIG_DIR = Path(__file__).parent
DEFAULT_CONFIG = CONFIG_DIR / "settings.yaml"


def load_config(path: Path = DEFAULT_CONFIG) -> dict:
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    for env_key, env_val in os.environ.items():
        if env_key.startswith("MEDAI__"):
            parts = env_key.lower().strip("medai_").split("__")
            node = config
            for part in parts[:-1]:
                node = node.setdefault(part, {})
            try:
                node[parts[-1]] = yaml.safe_load(env_val)
            except yaml.YAMLError:
                node[parts[-1]] = env_val
    return config


def setup_logging(config: dict) -> logging.Logger:
    lc = config.get("logging", {})
    level = getattr(logging, lc.get("level", "INFO"))
    fmt = lc.get("format", "%(asctime)s | %(name)s | %(levelname)s | %(message)s")
    logging.basicConfig(level=level, format=fmt, handlers=[logging.StreamHandler()])
    return logging.getLogger("medai")


@dataclass
class DatasetConfig:
    name: str = "diabetes_100k"
    path: str = "data/diabetes_100k.csv"
    target: str = "diabetes"
    categorical: list = field(default_factory=lambda: ["gender", "smoking_history"])
    numeric: list = field(
        default_factory=lambda: [
            "age",
            "hypertension",
            "heart_disease",
            "bmi",
            "HbA1c_level",
            "blood_glucose_level",
        ]
    )


@dataclass
class ModelConfig:
    algorithm: str = "lightgbm"
    n_estimators: int = 300
    max_depth: int = 6
    learning_rate: float = 0.05
    test_size: float = 0.2
    random_state: int = 42


@dataclass
class AppConfig:
    dataset: DatasetConfig
    model: ModelConfig
    raw: dict

    @classmethod
    def load(cls, path: Path = DEFAULT_CONFIG):
        cfg = load_config(path)
        return cls(
            dataset=DatasetConfig(**cfg.get("dataset", {})),
            model=ModelConfig(**cfg.get("model", {})),
            raw=cfg,
        )
