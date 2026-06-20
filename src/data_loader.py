"""
Enterprise data loader — multi-dataset support, real medical data.
"""

import logging
from pathlib import Path
from typing import List, Tuple

import pandas as pd

logger = logging.getLogger("medai.data")


class DataLoader:
    """Multi-dataset loader for real medical data."""

    SUPPORTED = {
        "diabetes_100k": {
            "path": "data/diabetes_100k.csv",
            "target": "diabetes",
            "categorical": ["gender", "smoking_history"],
        },
        "pima_diabetes": {
            "path": "data/pima_diabetes.csv",
            "target": "Outcome",
            "categorical": [],
        },
        "breast_cancer": {
            "path": "data/breast_cancer.csv",
            "target": "diagnosis",
            "categorical": [],
            "drop": ["id"],
            "encode_target": True,
        },
        "clinical_notes": {
            "path": "data/clinical_notes.csv",
            "target": "readmission_30d",
            "text_column": "discharge_note",
            "is_text": True,
        },
    }

    def __init__(self, dataset_name: str = "diabetes_100k", data_dir: str = "data"):
        self.dataset_name = dataset_name
        self.data_dir = Path(data_dir)
        if dataset_name not in self.SUPPORTED:
            raise ValueError(
                f"Unknown dataset '{dataset_name}'. Supported: {list(self.SUPPORTED.keys())}"
            )
        self.config = self.SUPPORTED[dataset_name]

    def load(self) -> pd.DataFrame:
        path = self.data_dir / self.config["path"]
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")
        df = pd.read_csv(path)
        if "drop" in self.config:
            df = df.drop(
                columns=[c for c in self.config["drop"] if c in df.columns], errors="ignore"
            )
        if self.config.get("encode_target"):
            df[self.config["target"]] = df[self.config["target"]].map({"M": 1, "B": 0})
        logger.info(f"Loaded {self.dataset_name}: {df.shape}")
        return df

    @property
    def target_col(self) -> str:
        return self.config["target"]

    @property
    def categorical_features(self) -> List[str]:
        return [c for c in self.config.get("categorical", []) if c != self.config["target"]]

    @property
    def numeric_features(self) -> List[str]:
        return [
            c for c in self.config.get("numeric", []) if c != self.config["target"]
        ] or None  # auto-detect if not specified

    def get_features(self, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        cat = [c for c in df.columns if c != self.target_col and df[c].dtype in ["object", "str"]]
        num = [
            c for c in df.columns if c != self.target_col and df[c].dtype in ["int64", "float64"]
        ]
        return cat, num

    def split(self, df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
        from sklearn.model_selection import train_test_split

        cat, num = self.get_features(df)
        feature_names = num + cat
        X = df[feature_names]
        y = df[self.target_col].values
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        logger.info(f"Train: {X_train.shape}, Test: {X_test.shape}, Features: {len(feature_names)}")
        return X_train, X_test, y_train, y_test, feature_names
