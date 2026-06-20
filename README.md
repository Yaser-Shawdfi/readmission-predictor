# 🏥 Patient Readmission Predictor — NLP on Clinical Notes

AI-powered 30-day hospital readmission prediction from clinical discharge summaries using NLP + Machine Learning.

## 📊 Project Overview

| Component | Details |
|-----------|---------|
| Task | Predict 30-day readmission risk from discharge notes |
| Dataset | 5,000 synthetic clinical discharge summaries |
| NLP | TF-IDF (500 features) + 11 clinical feature extractions |
| Models | Logistic Regression, XGBoost, LightGBM |
| Explainability | SHAP feature contributions |
| UI | Streamlit dashboard (5 pages) |

## 🧠 How It Works

```
┌──────────────┐     ┌───────────────┐     ┌──────────────┐     ┌─────────────┐
│  Clinical     │────▶│  NLP Pipeline  │────▶│  ML Model    │────▶│  Readmission │
│  Discharge    │     │  - TF-IDF      │     │  (XGBoost/   │     │  Risk Score  │
│  Note         │     │  - Feature     │     │   LightGBM/  │     │  + SHAP      │
│               │     │    Extraction  │     │   LogReg)    │     │  Explanation │
└──────────────┘     └───────────────┘     └──────────────┘     └─────────────┘
```

1. **Input**: Clinical discharge summary (free text)
2. **NLP**: TF-IDF vectorization (500 features) + structured feature extraction (11 clinical features)
3. **Model**: Ensemble of ML models predicts readmission probability
4. **Output**: Risk score (0-100%) + SHAP explanation of key risk factors

## 🚀 Quick Start

```bash
# Install
pip install -r requirements.txt

# Generate synthetic data
python src/generate_data.py

# Train models
PYTHONPATH=. python src/model.py

# Launch dashboard
streamlit run app/app.py --server.port 8501
```

## 📁 Project Structure

```
readmission-predictor/
├── config.py              # Configuration
├── requirements.txt       # Dependencies
├── src/
│   ├── generate_data.py   # Synthetic clinical notes generator
│   ├── nlp_pipeline.py    # TF-IDF + clinical feature extraction
│   └── model.py           # ML training (XGBoost, LightGBM, LogReg)
├── app/
│   └── app.py             # 5-page Streamlit dashboard
├── data/                  # Generated dataset
└── models/                # Saved model artifacts
```

## 📝 Dashboard Pages

1. **📊 Dashboard** — Overview metrics, readmission distribution, model performance
2. **📝 Predict from Note** — Paste clinical note → get risk prediction + SHAP explanation
3. **📈 Model Comparison** — Compare ROC-AUC, F1, precision, recall across models
4. **🔍 Feature Importance** — Top features driving readmission predictions
5. **📋 Data Explorer** — Browse clinical notes by readmission status

## 📄 License

MIT