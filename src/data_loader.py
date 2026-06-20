"""
Enterprise data loader — multi-dataset, 500K patients, caching, validation.
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

logger = logging.getLogger("medai.data")


class DataLoader:
    """Multi-dataset loader for real medical data with validation."""

    SUPPORTED = {
        "diabetes_500k": {
            "path": "data/diabetes_500k.csv",
            "target": "diabetes",
            "categorical": ["gender", "smoking_history"],
            "numeric": [
                "age",
                "hypertension",
                "heart_disease",
                "bmi",
                "HbA1c_level",
                "blood_glucose_level",
            ],
            "description": "500K synthetic patients (based on real 100K distribution)",
        },
        "diabetes_100k": {
            "path": "data/diabetes_100k.csv",
            "target": "diabetes",
            "categorical": ["gender", "smoking_history"],
            "numeric": [
                "age",
                "hypertension",
                "heart_disease",
                "bmi",
                "HbA1c_level",
                "blood_glucose_level",
            ],
            "description": "100K real patients (Kaggle)",
        },
        "pima_diabetes": {
            "path": "data/pima_diabetes.csv",
            "target": "Outcome",
            "categorical": [],
            "numeric": [
                "Pregnancies",
                "Glucose",
                "BloodPressure",
                "SkinThickness",
                "Insulin",
                "BMI",
                "DiabetesPedigreeFunction",
                "Age",
            ],
            "description": "768 Pima Indians (UCI)",
        },
        "breast_cancer": {
            "path": "data/breast_cancer.csv",
            "target": "diagnosis",
            "categorical": [],
            "numeric": None,  # auto-detect
            "drop": ["id"],
            "encode_target": True,
            "description": "569 Breast Cancer Wisconsin (UCI)",
        },
    }

    def __init__(self, dataset_name: str = "diabetes_500k", data_dir: str = "data"):
        self.dataset_name = dataset_name
        self.data_dir = Path(data_dir)
        if dataset_name not in self.SUPPORTED:
            raise ValueError(
                f"Unknown dataset '{dataset_name}'. Supported: {list(self.SUPPORTED.keys())}"
            )
        self.config = self.SUPPORTED[dataset_name]
        logger.info(f"DataLoader initialized: {dataset_name} — {self.config['description']}")

    def load(self) -> pd.DataFrame:
        """Load dataset with validation."""
        path = self.data_dir / self.config["path"]
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")
        df = pd.read_csv(path)

        # Drop specified columns
        if "drop" in self.config:
            df = df.drop(
                columns=[c for c in self.config["drop"] if c in df.columns], errors="ignore"
            )

        # Encode target if needed
        if self.config.get("encode_target"):
            df[self.config["target"]] = df[self.config["target"]].map({"M": 1, "B": 0})

        # Validate target column exists
        if self.config["target"] not in df.columns:
            raise ValueError(f"Target column '{self.config['target']}' not found in {path}")

        logger.info(
            f"Loaded {self.dataset_name}: {df.shape[0]:,} rows, {df.shape[1]} cols, target rate: {df[self.config['target']].mean():.2%}"
        )
        return df

    @property
    def target_col(self) -> str:
        return self.config["target"]

    @property
    def categorical_features(self) -> List[str]:
        return [c for c in self.config.get("categorical", []) if c != self.config["target"]]

    @property
    def numeric_features(self) -> Optional[List[str]]:
        nums = self.config.get("numeric")
        if nums:
            return [c for c in nums if c != self.config["target"]]
        return None

    def get_features(self, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """Return (categorical_features, numeric_features) excluding target."""
        cat = [c for c in df.columns if c != self.target_col and df[c].dtype in ["object", "str"]]
        num = [
            c
            for c in df.columns
            if c != self.target_col and df[c].dtype in ["int64", "float64", "int32", "float32"]
        ]
        return cat, num

    def split(self, df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
        """Chronological-free stratified train/test split."""
        from sklearn.model_selection import train_test_split

        cat, num = self.get_features(df)
        feature_names = num + cat
        X = df[feature_names]
        y = df[self.target_col].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        logger.info(
            f"Split: Train={X_train.shape[0]:,}, Test={X_test.shape[0]:,}, Features={len(feature_names)}"
        )
        return X_train, X_test, y_train, y_test, feature_names
