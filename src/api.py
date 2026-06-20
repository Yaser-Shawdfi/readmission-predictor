"""
FastAPI REST API for MedAI — 6 endpoints.
"""

import logging
from pathlib import Path
from typing import Optional

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
    description="Enterprise Medical AI — Diabetes & Readmission Prediction",
    version="2.0.0",
    docs_url="/docs",
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

MODELS_DIR = Path("models")
_trainer: Optional[ModelTrainer] = None
_loader: Optional[DataLoader] = None


def _get_trainer(dataset_name: str = "diabetes_100k") -> ModelTrainer:
    global _trainer, _loader
    if _trainer is None:
        _loader = DataLoader(dataset_name)
        df = _loader.load()
        X_train, X_test, y_train, y_test, features = _loader.split(df)
        _trainer = ModelTrainer()
        _trainer.train(X_train, X_test, y_train, y_test, features)
    return _trainer


class PredictRequest(BaseModel):
    dataset: str = "diabetes_100k"
    features: dict


class PredictResponse(BaseModel):
    prediction: int
    probability: float
    label: str


@app.get("/api/v1/health")
async def health():
    return {"status": "healthy", "version": "2.0.0"}


@app.get("/api/v1/datasets")
async def list_datasets():
    return {"datasets": list(DataLoader.SUPPORTED.keys())}


@app.get("/api/v1/datasets/{name}/info")
async def dataset_info(name: str):
    loader = DataLoader(name)
    df = loader.load()
    return {
        "name": name,
        "rows": len(df),
        "columns": list(df.columns),
        "target": loader.target_col,
        "categorical": loader.categorical_features,
    }


@app.post("/api/v1/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    trainer = _get_trainer(req.dataset)
    input_df = pd.DataFrame([req.features])
    result = trainer.predict(input_df)
    return PredictResponse(
        prediction=result["prediction"],
        probability=result["probability"],
        label="POSITIVE" if result["prediction"] == 1 else "NEGATIVE",
    )


@app.get("/api/v1/metrics")
async def get_metrics():
    path = MODELS_DIR / "model_comparison.csv"
    if path.exists():
        df = pd.read_csv(path)
        return df.to_dict(orient="records")
    raise HTTPException(404, "No metrics. Train a model first.")


@app.get("/api/v1/feature-importance")
async def feature_importance():
    trainer = _get_trainer()
    loader = DataLoader()
    df = loader.load()
    _, X_test, _, _, _ = loader.split(df)
    X_sample = trainer.preprocessor.transform(X_test[:200])
    X_sample = np.nan_to_num(X_sample.astype(np.float32), nan=-999.0)
    importance = trainer.explain_shap(X_sample, save_plot=False)
    return importance.to_dict(orient="records")


@app.get("/")
async def root():
    return {"name": "MedAI API", "version": "2.0.0", "docs": "/docs"}
