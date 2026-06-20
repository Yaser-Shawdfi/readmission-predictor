"""
Readmission prediction model training.
Trains Logistic Regression, XGBoost, and LightGBM on TF-IDF + clinical features.
"""

import numpy as np
import pandas as pd
import joblib
import logging
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
import xgboost as xgb
import lightgbm as lgb

logger = logging.getLogger("readmission.model")

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
RANDOM_STATE = 42


def get_models():
    return {
        "logistic_regression": LogisticRegression(
            random_state=RANDOM_STATE,
            max_iter=1000,
            class_weight="balanced",
            C=1.0,
        ),
        "xgboost": xgb.XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            eval_metric="auc",
            use_label_encoder=False,
        ),
        "lightgbm": lgb.LGBMClassifier(
            n_estimators=300,
            max_depth=6,
            num_leaves=31,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            verbose=-1,
            class_weight="balanced",
        ),
    }


def evaluate_model(model, X_test, y_test, name="model"):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    cm = confusion_matrix(y_test, y_pred)

    return {
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
    }


def train_and_compare(X_train, X_test, y_train, y_test, save=True):
    """Train all models, compare, save best."""
    models = get_models()
    all_metrics = []
    trained = {}

    for name, model in models.items():
        logger.info(f"Training {name}...")
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test, name)
        all_metrics.append(metrics)
        trained[name] = model
        logger.info(
            f"  ROC-AUC: {metrics['roc_auc']:.4f} | "
            f"F1: {metrics['f1']:.4f} | "
            f"Recall: {metrics['recall']:.4f}"
        )

    results = pd.DataFrame(all_metrics).sort_values("roc_auc", ascending=False).reset_index(drop=True)

    if save:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        best_name = results.iloc[0]["model"]
        joblib.dump(trained[best_name], MODELS_DIR / "best_model.joblib")
        results.to_csv(MODELS_DIR / "model_comparison.csv", index=False)
        logger.info(f"Best: {best_name} (ROC-AUC: {results.iloc[0]['roc_auc']:.4f})")

    return results, trained


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    from nlp_pipeline import prepare_data

    data_path = Path(__file__).resolve().parent.parent / "data" / "clinical_notes.csv"
    df = pd.read_csv(data_path)

    X_train, X_test, y_train, y_test, features, vec = prepare_data(df)
    results, models = train_and_compare(X_train, X_test, y_train, y_test)

    print(f"\n{'='*60}")
    print("MODEL COMPARISON")
    print(f"{'='*60}")
    print(results[["model", "accuracy", "precision", "recall", "f1", "roc_auc"]].to_string(index=False))