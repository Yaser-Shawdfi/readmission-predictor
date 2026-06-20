"""Tests for data loader module."""

import numpy as np
import pandas as pd
import pytest

from src.data_loader import DataLoader


def test_supported_datasets():
    assert "diabetes_100k" in DataLoader.SUPPORTED
    assert "breast_cancer" in DataLoader.SUPPORTED
    assert "pima_diabetes" in DataLoader.SUPPORTED


def test_unknown_dataset_raises():
    with pytest.raises(ValueError):
        DataLoader("nonexistent_dataset")


def test_target_col():
    loader = DataLoader("diabetes_100k")
    assert loader.target_col == "diabetes"


def test_get_features():
    dates = pd.date_range("2024-01-01", periods=100, freq="D")
    df = pd.DataFrame(
        {
            "age": np.random.randint(20, 80, 100).astype(np.int64),
            "bmi": np.random.uniform(15, 40, 100).astype(np.float64),
            "gender": np.random.choice(["M", "F"], 100),
            "diabetes": np.random.randint(0, 2, 100).astype(np.int64),
        },
        index=dates,
    )
    loader = DataLoader("diabetes_100k")
    cat, num = loader.get_features(df)
    assert "gender" in cat
    assert "age" in num
    assert "bmi" in num
    assert "diabetes" not in cat
    assert "diabetes" not in num
