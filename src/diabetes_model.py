"""
Diabetes Prediction Model — Real clinical data (100K patients).
Dataset: Kaggle Diabetes Prediction Dataset (iammustafatz/diabetes-prediction-dataset)
Features: gender, age, hypertension, heart_disease, smoking_history, bmi, HbA1c, blood_glucose
Target: diabetes (0/1)
"""

import pandas as pd
import numpy as np
import joblib
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
)
import xgboost as xgb
import lightgbm as lgb
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger("diabetes.model")

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"
RANDOM_STATE = 42


def load_diabetes_data():
    """Load the real diabetes prediction dataset."""
    filepath = DATA_DIR / "diabetes_prediction_dataset.csv"
    if not filepath.exists():
        raise FileNotFoundError(
            f"Dataset not found at {filepath}. "
            "Download from Kaggle: iammustafatz/diabetes-prediction-dataset"
        )
    df = pd.read_csv(filepath)
    logger.info(f"Loaded {len(df):,} patients, {len(df.columns)} features")
    return df


def get_categorical_features(df):
    return list(df.select_dtypes(include=["object", "str"]).columns)


def get_numeric_features(df, target="diabetes"):
    return [c for c in df.select_dtypes(include=[np.number]).columns if c != target]


def preprocess(df, test_size=0.2):
    """Preprocess: encode categoricals, split train/test."""
    cat_features = get_categorical_features(df)
    num_features = get_numeric_features(df)
    feature_names = num_features + cat_features

    X = df.drop(columns=["diabetes"])
    y = df["diabetes"].values
    X = X[feature_names]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_features),
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value",
                                   unknown_value=-1), cat_features),
        ],
    )

    X_train_p = preprocessor.fit_transform(X_train)
    X_test_p = preprocessor.transform(X_test)
    X_train_p = np.nan_to_num(X_train_p.astype(np.float32), nan=-999)
    X_test_p = np.nan_to_num(X_test_p.astype(np.float32), nan=-999)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, MODELS_DIR / "diabetes_preprocessor.joblib")

    logger.info(f"Train: {X_train_p.shape}, Test: {X_test_p.shape}")
    logger.info(f"Features: {feature_names}")
    logger.info(f"Train diabetes rate: {y_train.mean():.2%}")

    return X_train_p, X_test_p, y_train, y_test, feature_names, preprocessor


def get_models():
    return {
        "logistic_regression": LogisticRegression(
            random_state=RANDOM_STATE, max_iter=1000, C=1.0,
        ),
        "xgboost": xgb.XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=RANDOM_STATE, eval_metric="auc",
        ),
        "lightgbm": lgb.LGBMClassifier(
            n_estimators=300, max_depth=6, num_leaves=31,
            learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
            random_state=RANDOM_STATE, verbose=-1,
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
        "tn": int(cm[0, 0]), "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]), "tp": int(cm[1, 1]),
    }


def train_and_compare(X_train, X_test, y_train, y_test, save=True):
    models = get_models()
    all_metrics = []
    trained = {}

    for name, model in models.items():
        logger.info(f"Training {name}...")
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test, name)
        all_metrics.append(metrics)
        trained[name] = model
        logger.info(f"  ROC-AUC: {metrics['roc_auc']:.4f} | F1: {metrics['f1']:.4f} | Recall: {metrics['recall']:.4f}")

    results = pd.DataFrame(all_metrics).sort_values("roc_auc", ascending=False).reset_index(drop=True)

    if save:
        best_name = results.iloc[0]["model"]
        joblib.dump(trained[best_name], MODELS_DIR / "diabetes_best_model.joblib")
        results.to_csv(MODELS_DIR / "diabetes_model_comparison.csv", index=False)
        logger.info(f"Best: {best_name} (ROC-AUC: {results.iloc[0]['roc_auc']:.4f})")

    return results, trained


def compute_shap_importance(model, X_train, X_sample, feature_names, save_plot=True):
    """Compute SHAP global feature importance."""
    try:
        explainer = shap.TreeExplainer(model)
    except Exception:
        explainer = shap.Explainer(model, X_train[:1000])

    shap_values = explainer.shap_values(X_sample)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]

    mean_abs = np.abs(shap_values).mean(axis=0)
    importance = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    if save_plot:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.summary_plot(shap_values, X_sample, feature_names=feature_names,
                          plot_type="bar", show=False, max_display=10)
        plt.title("Diabetes Prediction — Global Feature Importance (SHAP)")
        plt.tight_layout()
        plt.savefig(REPORTS_DIR / "diabetes_shap_importance.png", dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved SHAP plot to {REPORTS_DIR / 'diabetes_shap_importance.png'}")

    return importance


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    df = load_diabetes_data()
    print(f"\nDataset: {df.shape}")
    print(f"Diabetes rate: {df['diabetes'].mean():.2%}")
    print(f"\nColumns: {list(df.columns)}")

    X_train, X_test, y_train, y_test, features, prep = preprocess(df)
    results, models = train_and_compare(X_train, X_test, y_train, y_test)

    print(f"\n{'='*60}")
    print("DIABETES PREDICTION — MODEL COMPARISON")
    print(f"{'='*60}")
    print(results[["model", "accuracy", "precision", "recall", "f1", "roc_auc"]].to_string(index=False))

    # SHAP on best model
    best_name = results.iloc[0]["model"]
    best = models[best_name]
    print(f"\nComputing SHAP importance for {best_name}...")
    importance = compute_shap_importance(best, X_train, X_test[:500], features)
    print("\nTop 10 features:")
    print(importance.head(10).to_string(index=False))