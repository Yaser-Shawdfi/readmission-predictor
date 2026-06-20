"""
Enterprise model trainer — multi-model, SHAP, full metrics.
Supports: diabetes, breast cancer, clinical notes (NLP).
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import lightgbm as lgb
import matplotlib
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import OrdinalEncoder

matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger("medai.model")

MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")


class ModelTrainer:
    """Train, evaluate, and explain multiple ML models."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.preprocessor: Optional[ColumnTransformer] = None
        self.models: Dict[str, Any] = {}
        self.results: Optional[pd.DataFrame] = None
        self.best_model = None
        self.best_name: str = ""
        self.feature_names: List[str] = []

    def _build_preprocessor(
        self, cat_features: List[str], num_features: List[str]
    ) -> ColumnTransformer:
        transformers = [("num", "passthrough", num_features)]
        if cat_features:
            transformers.append(
                (
                    "cat",
                    OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                    cat_features,
                )
            )
        return ColumnTransformer(transformers=transformers, remainder="drop")

    def _get_models(self) -> Dict[str, Any]:
        return {
            "logistic_regression": LogisticRegression(
                random_state=self.random_state,
                max_iter=1000,
                C=1.0,
            ),
            "xgboost": xgb.XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state,
                eval_metric="auc",
            ),
            "lightgbm": lgb.LGBMClassifier(
                n_estimators=300,
                max_depth=6,
                num_leaves=31,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state,
                verbose=-1,
            ),
        }

    def train(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: np.ndarray,
        y_test: np.ndarray,
        feature_names: List[str],
    ) -> Tuple[pd.DataFrame, Dict]:
        """Train all models, return comparison results + trained models."""
        cat_features = list(X_train.select_dtypes(include=["object", "str"]).columns)
        num_features = [c for c in X_train.columns if c not in cat_features]

        # Build and fit preprocessor
        transformer_specs = [("num", "passthrough", num_features)]
        if cat_features:
            transformer_specs.append(
                (
                    "cat",
                    OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                    cat_features,
                )
            )
        self.preprocessor = ColumnTransformer(transformers=transformer_specs, remainder="drop")
        X_train_p = self.preprocessor.fit_transform(X_train)
        X_test_p = self.preprocessor.transform(X_test)
        X_train_p = np.nan_to_num(X_train_p.astype(np.float32), nan=-999.0)
        X_test_p = np.nan_to_num(X_test_p.astype(np.float32), nan=-999.0)
        self.feature_names = feature_names

        # Save preprocessor
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.preprocessor, MODELS_DIR / "preprocessor.joblib")

        # Train models
        models = self._get_models()
        all_metrics = []
        for name, model in models.items():
            logger.info(f"Training {name}...")
            model.fit(X_train_p, y_train)
            metrics = self._evaluate(model, X_test_p, y_test, name)
            all_metrics.append(metrics)
            self.models[name] = model
            logger.info(f"  ROC-AUC: {metrics['roc_auc']:.4f} | F1: {metrics['f1']:.4f}")

        self.results = (
            pd.DataFrame(all_metrics).sort_values("roc_auc", ascending=False).reset_index(drop=True)
        )
        self.best_name = self.results.iloc[0]["model"]
        self.best_model = self.models[self.best_name]
        joblib.dump(self.best_model, MODELS_DIR / "best_model.joblib")
        self.results.to_csv(MODELS_DIR / "model_comparison.csv", index=False)

        logger.info(f"Best: {self.best_name} (ROC-AUC: {self.results.iloc[0]['roc_auc']:.4f})")
        return self.results, self.models

    def _evaluate(self, model, X_test, y_test, name) -> Dict:
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
            "avg_precision": average_precision_score(y_test, y_proba),
            "tn": int(cm[0, 0]),
            "fp": int(cm[0, 1]),
            "fn": int(cm[1, 0]),
            "tp": int(cm[1, 1]),
        }

    def predict(self, input_df: pd.DataFrame) -> Dict[str, Any]:
        """Predict on new data."""
        X = self.preprocessor.transform(input_df)
        X = np.nan_to_num(X.astype(np.float32), nan=-999.0)
        prediction = int(self.best_model.predict(X)[0])
        probability = float(self.best_model.predict_proba(X)[0, 1])
        return {"prediction": prediction, "probability": probability}

    def explain_shap(self, X_sample: np.ndarray, save_plot: bool = True) -> pd.DataFrame:
        """Compute SHAP global feature importance."""
        try:
            explainer = shap.TreeExplainer(self.best_model)
        except Exception:
            explainer = shap.Explainer(self.best_model, X_sample[:100])

        shap_values = explainer.shap_values(X_sample)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            shap_values = shap_values[:, :, 1]

        mean_abs = np.abs(shap_values).mean(axis=0)
        importance = (
            pd.DataFrame(
                {
                    "feature": self.feature_names,
                    "mean_abs_shap": mean_abs,
                }
            )
            .sort_values("mean_abs_shap", ascending=False)
            .reset_index(drop=True)
        )

        if save_plot:
            REPORTS_DIR.mkdir(parents=True, exist_ok=True)
            fig, ax = plt.subplots(figsize=(10, 6))
            shap.summary_plot(
                shap_values,
                X_sample,
                feature_names=self.feature_names,
                plot_type="bar",
                show=False,
                max_display=15,
            )
            plt.title("Global Feature Importance (SHAP)")
            plt.tight_layout()
            plt.savefig(REPORTS_DIR / "shap_importance.png", dpi=150, bbox_inches="tight")
            plt.close()

        return importance

    def explain_prediction_shap(self, input_df: pd.DataFrame, save_plot: bool = True) -> Dict:
        """Explain a single prediction with SHAP waterfall."""
        X = self.preprocessor.transform(input_df)
        X = np.nan_to_num(X.astype(np.float32), nan=-999.0)

        try:
            explainer = shap.TreeExplainer(self.best_model)
        except Exception:
            explainer = shap.Explainer(self.best_model, X)

        shap_values = explainer.shap_values(X)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        sv = shap_values[0] if shap_values.ndim > 1 else shap_values

        base = explainer.expected_value
        if isinstance(base, (list, np.ndarray)):
            base = base[1]

        contributions = (
            pd.DataFrame(
                {
                    "feature": self.feature_names,
                    "shap_value": sv,
                    "direction": ["↑ increases risk" if v > 0 else "↓ decreases risk" for v in sv],
                }
            )
            .assign(abs_shap=lambda x: x.shap_value.abs())
            .sort_values("abs_shap", ascending=False)
        )

        if save_plot:
            REPORTS_DIR.mkdir(parents=True, exist_ok=True)
            fig, ax = plt.subplots(figsize=(10, 6))
            shap.waterfall_plot(
                shap.Explanation(
                    values=sv, base_values=base, data=X[0], feature_names=self.feature_names
                ),
                max_display=15,
                show=False,
            )
            plt.tight_layout()
            plt.savefig(REPORTS_DIR / "shap_waterfall.png", dpi=150, bbox_inches="tight")
            plt.close()

        return {"base_value": float(base), "contributions": contributions}
