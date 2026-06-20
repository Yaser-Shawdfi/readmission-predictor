# 🏥 MedAI v3 — Enterprise Medical AI

AI-powered diabetes prediction on **500,000 patients** with SHAP explainability, FastAPI REST API, CLI, Docker, and CI/CD.

## 📊 Overview

| Component | Details |
|-----------|---------|
| Dataset | 500,000 patients (synthetic, based on real 100K distribution) |
| Target | Diabetes (binary, 8.79% positive rate) |
| Features | 8 clinical features (HbA1c, blood glucose, BMI, age, etc.) |
| Models | XGBoost, LightGBM, Logistic Regression |
| Best ROC-AUC | 0.863 (XGBoost) |
| Explainability | SHAP global + local waterfall |
| API | FastAPI (7 endpoints + Swagger) |
| Tests | 10/10 passing (pytest) |
| Deployment | Docker + docker-compose |
| CI/CD | GitHub Actions (lint → test → Docker build) |

## 🏗️ Architecture

```
readmission-predictor/
├── pyproject.toml              # PEP 621 package (pip install -e .)
├── src/
│   ├── cli.py                  # CLI: medai train/predict/api/ui/datasets
│   ├── config.py               # YAML config + dataclasses + env overrides
│   ├── settings.yaml           # All config in one file
│   ├── api.py                  # FastAPI REST (7 endpoints + Swagger docs)
│   ├── data_loader.py          # Multi-dataset loader (4 datasets)
│   ├── model_trainer.py        # Train/evaluate/explain (XGBoost, LightGBM, LogReg)
│   ├── generate_data.py        # Synthetic clinical notes generator (legacy)
│   ├── nlp_pipeline.py         # TF-IDF + clinical features (legacy)
│   ├── diabetes_model.py       # Original diabetes model (legacy)
│   └── model.py                # Original readmission model (legacy)
├── app/app.py                  # 6-page Streamlit dashboard
├── tests/                      # 10 unit tests (all passing)
├── Dockerfile                  # Multi-stage Docker build
├── docker-compose.yml          # API + UI services
├── .github/workflows/ci.yml    # GitHub Actions CI/CD
└── scripts/create_test_data.py # CI fallback data generator
```

## 🚀 Quick Start

### Install
```bash
pip install -e ".[dev]"
```

### CLI Commands
```bash
medai datasets                           # List supported datasets
medai train --dataset diabetes_500k      # Train models
medai predict --dataset diabetes_500k \
  --features '{"age":55,"bmi":32,"HbA1c_level":7.2,...}'
medai api --port 8000                     # Start REST API
medai ui --port 8501                      # Start Streamlit dashboard
```

### REST API
```bash
# Start
uvicorn src.api:app --port 8000

# Health check
curl http://localhost:8000/api/v1/health

# Predict
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"dataset":"diabetes_500k","features":{"gender":"Male","age":55,"hypertension":1,"heart_disease":0,"smoking_history":"current","bmi":32.5,"HbA1c_level":7.2,"blood_glucose_level":180}}'

# Swagger docs: http://localhost:8000/docs
```

### Docker
```bash
docker-compose up          # API + UI
docker-compose run test    # Run tests
```

### Tests
```bash
pytest tests/ -v            # 10/10 passing
```

## 📊 Model Results (500K Patients)

| Model | ROC-AUC | F1 | Precision | Recall | Avg Precision |
|-------|---------|-----|-----------|--------|---------------|
| XGBoost | **0.8631** | 0.3846 | 0.7638 | 0.2570 | — |
| LightGBM | 0.8629 | 0.3848 | 0.7653 | 0.2570 | — |
| LogReg | 0.8032 | 0.1474 | 0.5829 | 0.0844 | — |

## 📄 License

MIT