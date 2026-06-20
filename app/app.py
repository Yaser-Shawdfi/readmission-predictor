"""
Streamlit Dashboard — MedAI v3 Enterprise (500K patients).
6 pages: Overview, Diabetes Predictor, Model Comparison, Feature Importance, API docs, Data Explorer.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

from src.data_loader import DataLoader
from src.model_trainer import ModelTrainer

MODELS_DIR = PROJECT_ROOT / "models"

st.set_page_config(page_title="MedAI v3", page_icon="🏥", layout="wide")


@st.cache_data
def load_data(dataset_name="diabetes_500k"):
    loader = DataLoader(dataset_name)
    return loader.load()


@st.cache_resource
def get_pipeline(dataset_name="diabetes_500k"):
    loader = DataLoader(dataset_name)
    df = loader.load()
    X_train, X_test, y_train, y_test, features = loader.split(df)
    trainer = ModelTrainer()
    results, models = trainer.train(X_train, X_test, y_train, y_test, features)
    return df, X_train, X_test, y_train, y_test, features, results, trainer


with st.spinner("Loading 500K patient dataset and training models..."):
    df, X_train, X_test, y_train, y_test, feature_names, results, trainer = get_pipeline("diabetes_500k")


st.sidebar.title("🏥 MedAI v3")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate", [
    "📊 Overview",
    "🩺 Diabetes Predictor",
    "📈 Model Comparison",
    "🔍 Feature Importance",
    "📋 Data Explorer",
    "🔌 API Reference",
])
st.sidebar.markdown("---")
st.sidebar.markdown("**Enterprise MedAI v3**")
st.sidebar.markdown(f"**Dataset:** 500,000 patients")
st.sidebar.markdown(f"**Best model:** {results.iloc[0]['model']}")
st.sidebar.markdown(f"**ROC-AUC:** {results.iloc[0]['roc_auc']:.4f}")


def risk_gauge(probability, title="Risk Score"):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=probability * 100,
        domain={"x": [0, 1], "y": [0, 1]}, title={"text": title},
        gauge={
            "axis": {"range": [0, 100]}, "bar": {"color": "#1f77b4"},
            "steps": [
                {"range": [0, 30], "color": "#2ecc71"},
                {"range": [30, 60], "color": "#f39c12"},
                {"range": [60, 100], "color": "#e74c3c"},
            ],
            "threshold": {"line": {"color": "black", "width": 4}, "thickness": 0.75, "value": 50},
        },
    ))
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
    return fig


# ─── Overview ────────────────────────────────────────────────────────────────
if page == "📊 Overview":
    st.title("📊 MedAI v3 — Enterprise Medical AI")
    st.markdown("AI-powered diabetes prediction trained on **500,000 patients** with SHAP explainability.")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Patients", f"{len(df):,}")
    col2.metric("Diabetes Rate", f"{df['diabetes'].mean():.1%}")
    col3.metric("Features", len(feature_names))
    col4.metric("Best Model", results.iloc[0]["model"].replace("_", " ").title())

    st.metric("🏆 Best ROC-AUC", f"{results.iloc[0]['roc_auc']:.4f}")

    st.markdown("---")
    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("Diabetes Distribution")
        counts = df["diabetes"].value_counts().rename({0: "No Diabetes", 1: "Diabetes"})
        fig = px.bar(x=counts.index, y=counts.values, color=counts.index,
                     color_discrete_map={"No Diabetes": "#2ecc71", "Diabetes": "#e74c3c"},
                     labels={"x": "", "y": "Count"})
        fig.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.subheader("Model Performance (ROC-AUC)")
        fig = px.bar(results, x="model", y="roc_auc", color="model",
                     labels={"model": "Model", "roc_auc": "ROC-AUC"}, range_y=[0.5, 1.0])
        fig.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Model Metrics")
    st.dataframe(results[["model", "accuracy", "precision", "recall", "f1", "roc_auc", "avg_precision"]].style.format({
        "accuracy": "{:.4f}", "precision": "{:.4f}", "recall": "{:.4f}",
        "f1": "{:.4f}", "roc_auc": "{:.4f}", "avg_precision": "{:.4f}",
    }), use_container_width=True)

    st.markdown("---")
    st.subheader("Enterprise Features")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("✅ Python package (pyproject.toml)")
        st.markdown("✅ FastAPI REST API (7 endpoints)")
        st.markdown("✅ CLI tool (medai command)")
    with col2:
        st.markdown("✅ 10 unit tests (pytest)")
        st.markdown("✅ Docker + docker-compose")
        st.markdown("✅ CI/CD (GitHub Actions)")
    with col3:
        st.markdown("✅ YAML config + env overrides")
        st.markdown("✅ Structured logging")
        st.markdown("✅ SHAP explainability")


# ─── Diabetes Predictor ─────────────────────────────────────────────────────
elif page == "🩺 Diabetes Predictor":
    st.title("🩺 Diabetes Risk Predictor")
    st.markdown("Enter patient vitals → get diabetes risk prediction with SHAP explanation. **Trained on 500K patients.**")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Patient Vitals")
        age = st.slider("Age", 0, 100, 50)
        bmi = st.slider("BMI", 10.0, 60.0, 27.0, 0.1)
        hba1c = st.slider("HbA1c Level (%)", 3.0, 15.0, 5.5, 0.1)
        glucose = st.slider("Blood Glucose Level (mg/dL)", 50, 300, 140)
        hypertension = st.selectbox("Hypertension", ["No", "Yes"])
        heart_disease = st.selectbox("Heart Disease", ["No", "Yes"])

    with col2:
        st.subheader("Demographics")
        gender = st.selectbox("Gender", ["Female", "Male", "Other"])
        smoking = st.selectbox("Smoking History", [
            "never", "No Info", "current", "former", "not current", "ever"
        ])

    if st.button("🚀 Predict Diabetes Risk", type="primary"):
        with st.spinner("Running prediction on 500K-trained model..."):
            input_df = pd.DataFrame([{
                "gender": gender,
                "age": float(age),
                "hypertension": 1 if hypertension == "Yes" else 0,
                "heart_disease": 1 if heart_disease == "Yes" else 0,
                "smoking_history": smoking,
                "bmi": bmi,
                "HbA1c_level": hba1c,
                "blood_glucose_level": glucose,
            }])

            result = trainer.predict(input_df)
            prob = result["probability"]
            risk = "low" if prob < 0.3 else ("medium" if prob < 0.6 else "high")

            col_res, col_gauge = st.columns([1, 1])
            with col_res:
                if result["prediction"] == 1:
                    st.error(f"⚠️ HIGH RISK — Diabetes Likely")
                    st.markdown(f"**Diabetes probability: {prob:.1%}**")
                    st.markdown("Recommend: Confirm with fasting glucose test + HbA1c. Refer to endocrinologist.")
                else:
                    st.success(f"✅ LOW RISK — Diabetes Unlikely")
                    st.markdown(f"**Diabetes probability: {prob:.1%}**")
                    st.markdown("Routine screening recommended at next checkup.")
                st.markdown(f"**Risk level: {risk.upper()}**")
            with col_gauge:
                st.plotly_chart(risk_gauge(prob, "Diabetes Risk Score"), use_container_width=True)

            # SHAP explanation
            st.markdown("---")
            st.subheader("🔍 SHAP Explanation — Key Risk Factors")
            try:
                shap_result = trainer.explain_prediction_shap(input_df, save_plot=False)
                contributions = shap_result["contributions"]

                fig, ax = plt.subplots(figsize=(10, 5))
                X_processed = trainer.preprocessor.transform(input_df)
                X_processed = np.nan_to_num(X_processed.astype(np.float32), nan=-999.0)

                explainer = shap.TreeExplainer(trainer.best_model)
                shap_values = explainer.shap_values(X_processed)
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]
                sv = shap_values[0] if shap_values.ndim > 1 else shap_values
                base = explainer.expected_value
                if isinstance(base, (list, np.ndarray)):
                    base = base[1]

                shap.waterfall_plot(
                    shap.Explanation(values=sv, base_values=base, data=X_processed[0],
                                     feature_names=feature_names),
                    max_display=8, show=False,
                )
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close()

                st.dataframe(contributions[["feature", "shap_value", "direction"]].head(8).style.format({
                    "shap_value": "{:.4f}"
                }), use_container_width=True)
            except Exception as e:
                st.info(f"SHAP explanation: {e}")


# ─── Model Comparison ───────────────────────────────────────────────────────
elif page == "📈 Model Comparison":
    st.title("📈 Model Comparison")
    st.markdown("Performance across all trained models on 500K patients.")

    st.subheader("Metrics")
    display_df = results[["model", "accuracy", "precision", "recall", "f1", "roc_auc", "avg_precision", "brier_score"]].copy()
    display_df["model"] = display_df["model"].str.replace("_", " ").str.title()
    st.dataframe(display_df.style.format({
        "accuracy": "{:.4f}", "precision": "{:.4f}", "recall": "{:.4f}",
        "f1": "{:.4f}", "roc_auc": "{:.4f}", "avg_precision": "{:.4f}",
        "brier_score": "{:.4f}",
    }), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("ROC-AUC Comparison")
        fig = px.bar(results, x="model", y="roc_auc", color="model", range_y=[0.5, 1.0])
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("F1 Score Comparison")
        fig = px.bar(results, x="model", y="f1", color="model")
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)


# ─── Feature Importance ─────────────────────────────────────────────────────
elif page == "🔍 Feature Importance":
    st.title("🔍 Feature Importance")
    st.markdown("Which features drive diabetes predictions?")

    if hasattr(trainer.best_model, "feature_importances_"):
        imp = trainer.best_model.feature_importances_
        imp_df = pd.DataFrame({"Feature": feature_names, "Importance": imp}).sort_values("Importance", ascending=False)

        st.subheader("Top Features (Tree Importance)")
        fig = px.bar(imp_df, x="Importance", y="Feature", orientation="h",
                     color="Importance", color_continuous_scale="Viridis")
        fig.update_layout(height=400, yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)
    elif hasattr(trainer.best_model, "coef_"):
        coef = trainer.best_model.coef_[0]
        imp_df = pd.DataFrame({"Feature": feature_names, "Coefficient": coef}).assign(
            Abs=lambda d: d.Coefficient.abs()
        ).sort_values("Abs", ascending=False)

        st.subheader("Top Features (Logistic Coefficients)")
        fig = px.bar(imp_df, x="Abs", y="Feature", orientation="h",
                     color="Abs", color_continuous_scale="Viridis")
        fig.update_layout(height=400, yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)

    # SHAP global
    st.markdown("---")
    st.subheader("SHAP Global Importance")
    with st.spinner("Computing SHAP values..."):
        X_sample = trainer.preprocessor.transform(X_test[:500])
        X_sample = np.nan_to_num(X_sample.astype(np.float32), nan=-999.0)
        importance = trainer.explain_shap(X_sample, save_plot=False)
        st.dataframe(importance.style.format({"mean_abs_shap": "{:.4f}"}), use_container_width=True)


# ─── Data Explorer ──────────────────────────────────────────────────────────
elif page == "📋 Data Explorer":
    st.title("📋 Data Explorer — 500K Patients")
    st.markdown("Explore the 500,000-patient diabetes dataset.")

    st.metric("Total patients", f"{len(df):,}")
    st.metric("Diabetes rate", f"{df['diabetes'].mean():.1%}")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Age Distribution by Diabetes")
        sample = df.sample(min(10000, len(df)))
        fig = px.histogram(sample, x="age", color="diabetes", nbins=50,
                           color_discrete_map={0: "#2ecc71", 1: "#e74c3c"},
                           labels={"diabetes": "Diabetes"})
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("HbA1c vs Blood Glucose")
        fig = px.scatter(sample, x="HbA1c_level", y="blood_glucose_level", color="diabetes",
                         color_discrete_map={0: "#2ecc71", 1: "#e74c3c"},
                         labels={"diabetes": "Diabetes"}, opacity=0.5)
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Feature Distributions")
    feature = st.selectbox("Select feature", [c for c in df.columns if c != "diabetes"])
    if df[feature].dtype in ["object", "str"]:
        counts = df[feature].value_counts().head(20)
        fig = px.bar(x=counts.index, y=counts.values, labels={"x": feature, "y": "Count"})
    else:
        fig = px.histogram(df, x=feature, color="diabetes", nbins=50,
                           color_discrete_map={0: "#2ecc71", 1: "#e74c3c"})
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Sample Data")
    st.dataframe(df.head(20), use_container_width=True)


# ─── API Reference ───────────────────────────────────────────────────────────
elif page == "🔌 API Reference":
    st.title("🔌 API Reference")
    st.markdown("MedAI v3 exposes a REST API with 7 endpoints. Start the server with `medai api`.")

    st.code("""
# Start the API server
medai api --port 8000

# Or via uvicorn
uvicorn src.api:app --port 8000
""", language="bash")

    st.markdown("### Endpoints")
    st.markdown("""
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/health` | Health check |
| GET | `/api/v1/datasets` | List all datasets |
| GET | `/api/v1/datasets/{name}/info` | Dataset metadata + sample |
| POST | `/api/v1/predict` | Predict diabetes risk |
| GET | `/api/v1/metrics` | Model comparison metrics |
| GET | `/api/v1/feature-importance` | SHAP feature importance |
| GET | `/docs` | Interactive Swagger UI |
""")

    st.markdown("### Example: Predict")
    st.code("""
curl -X POST http://localhost:8000/api/v1/predict \\
  -H "Content-Type: application/json" \\
  -d '{
    "dataset": "diabetes_500k",
    "features": {
      "gender": "Male",
      "age": 55,
      "hypertension": 1,
      "heart_disease": 0,
      "smoking_history": "current",
      "bmi": 32.5,
      "HbA1c_level": 7.2,
      "blood_glucose_level": 180
    }
  }'
""", language="bash")

    st.markdown("### Response")
    st.code("""
{
  "prediction": 1,
  "probability": 0.78,
  "label": "POSITIVE",
  "risk_level": "high"
}
""", language="json")