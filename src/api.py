"""
FastAPI REST API — 7 endpoints, lazy model loading, OpenAPI docs.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.data_loader import DataLoader
from src.model_trainer import ModelTrainer

logger = logging.getLogger("medai.api")

app = FastAPI(
    title="MedAI API",
    description="Enterprise Medical AI — Diabetes Prediction (500K patients)",
    version="3.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODELS_DIR = Path("models")
_trainer: Optional[ModelTrainer] = None
_loader: Optional[DataLoader] = None
_data_cache: Dict[str, pd.DataFrame] = {}


def _get_data(dataset_name: str) -> pd.DataFrame:
    if dataset_name not in _data_cache:
        loader = DataLoader(dataset_name)
        _data_cache[dataset_name] = loader.load()
    return _data_cache[dataset_name]


def _get_trainer(dataset_name: str = "diabetes_500k") -> ModelTrainer:
    global _trainer, _loader
    if _trainer is None:
        _loader = DataLoader(dataset_name)
        df = _loader.load()
        X_train, X_test, y_train, y_test, features = _loader.split(df)
        _trainer = ModelTrainer()
        _trainer.train(X_train, X_test, y_train, y_test, features)
    return _trainer


class PredictRequest(BaseModel):
    dataset: str = "diabetes_500k"
    features: Dict[str, Any]


class PredictResponse(BaseModel):
    prediction: int
    probability: float
    label: str
    risk_level: str


class TrainRequest(BaseModel):
    dataset: str = "diabetes_500k"


@app.get("/api/v1/health")
async def health():
    """Health check."""
    return {"status": "healthy", "version": "3.0.0", "model_loaded": _trainer is not None}


@app.get("/")
async def root():
    return {"name": "MedAI API", "version": "3.0.0", "docs": "/docs"}


@app.get("/api/v1/datasets")
async def list_datasets():
    """List all supported datasets."""
    result = {}
    for name, config in DataLoader.SUPPORTED.items():
        result[name] = {
            "description": config.get("description", ""),
            "target": config["target"],
            "features": len(config.get("numeric", []) or []) + len(config.get("categorical", [])),
        }
    return {"datasets": result}


@app.get("/api/v1/datasets/{name}/info")
async def dataset_info(name: str):
    """Get dataset metadata + sample."""
    if name not in DataLoader.SUPPORTED:
        raise HTTPException(404, f"Dataset '{name}' not found")
    loader = DataLoader(name)
    df = loader.load()
    cat, num = loader.get_features(df)
    return {
        "name": name,
        "rows": len(df),
        "columns": list(df.columns),
        "target": loader.target_col,
        "categorical": cat,
        "numeric": num,
        "target_rate": float(df[loader.target_col].mean()),
        "sample": df.head(5).to_dict(orient="records"),
    }


@app.post("/api/v1/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    """Predict disease risk from patient features."""
    trainer = _get_trainer(req.dataset)
    input_df = pd.DataFrame([req.features])
    result = trainer.predict(input_df)
    prob = result["probability"]
    risk = "low" if prob < 0.3 else ("medium" if prob < 0.6 else "high")
    return PredictResponse(
        prediction=result["prediction"],
        probability=prob,
        label="POSITIVE" if result["prediction"] == 1 else "NEGATIVE",
        risk_level=risk,
    )


@app.get("/api/v1/metrics")
async def get_metrics():
    """Get model comparison metrics."""
    path = MODELS_DIR / "model_comparison.csv"
    if path.exists():
        df = pd.read_csv(path)
        return df.to_dict(orient="records")
    raise HTTPException(404, "No metrics found. Train a model first.")


@app.get("/api/v1/feature-importance")
async def feature_importance(dataset: str = "diabetes_500k"):
    """Get SHAP feature importance."""
    trainer = _get_trainer(dataset)
    loader = DataLoader(dataset)
    df = loader.load()
    _, X_test, _, _, _ = loader.split(df)
    X_sample = trainer.preprocessor.transform(X_test[:500])
    X_sample = np.nan_to_num(X_sample.astype(np.float32), nan=-999.0)
    importance = trainer.explain_shap(X_sample, save_plot=False)
    return importance.to_dict(orient="records")
