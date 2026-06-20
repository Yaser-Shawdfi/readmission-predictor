"""Tests for model trainer module."""

import numpy as np
import pandas as pd
import pytest

from src.model_trainer import ModelTrainer


@pytest.fixture
def sample_data():
    np.random.seed(42)
    n = 200
    df = pd.DataFrame(
        {
            "age": np.random.randint(20, 80, n),
            "bmi": np.random.uniform(15, 40, n),
            "glucose": np.random.randint(70, 200, n),
            "gender": np.random.choice(["M", "F"], n),
        }
    )
    y = ((df["glucose"] > 140) & (df["age"] > 50)).astype(int).values
    return df, y


def test_trainer_init():
    trainer = ModelTrainer(random_state=42)
    assert trainer.random_state == 42
    assert trainer.best_model is None


def test_train(sample_data):
    df, y = sample_data
    X_train = df.iloc[:160]
    X_test = df.iloc[160:]
    y_train = y[:160]
    y_test = y[160:]

    trainer = ModelTrainer()
    results, models = trainer.train(X_train, X_test, y_train, y_test, list(df.columns))

    assert "roc_auc" in results.columns
    assert len(results) == 3  # 3 models
    assert trainer.best_model is not None


def test_predict(sample_data):
    df, y = sample_data
    trainer = ModelTrainer()
    trainer.train(df.iloc[:160], df.iloc[160:], y[:160], y[160:], list(df.columns))

    input_df = pd.DataFrame([{"age": 65, "bmi": 30, "glucose": 180, "gender": "M"}])
    result = trainer.predict(input_df)
    assert "prediction" in result
    assert "probability" in result
    assert 0 <= result["probability"] <= 1
