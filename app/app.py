"""
Streamlit Dashboard for Patient Readmission Predictor.
Paste a clinical discharge note → get readmission risk prediction + explanation.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

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

from config import MODELS_DIR
from nlp_pipeline import clean_clinical_text, extract_clinical_features, prepare_data
from model import train_and_compare


st.set_page_config(page_title="Readmission Predictor", page_icon="🏥", layout="wide")

MODELS_DIR = Path(PROJECT_ROOT) / "models"


@st.cache_data
def load_data():
    data_path = PROJECT_ROOT / "data" / "clinical_notes.csv"
    return pd.read_csv(data_path)


@st.cache_resource
def get_pipeline():
    df = load_data()
    X_train, X_test, y_train, y_test, features, vec = prepare_data(df)
    results, models = train_and_compare(X_train, X_test, y_train, y_test)
    best = joblib.load(MODELS_DIR / "best_model.joblib")
    return df, X_train, X_test, y_train, y_test, features, vec, results, best


with st.spinner("Loading data and training models..."):
    df, X_train, X_test, y_train, y_test, feature_names, vectorizer, results, best_model = get_pipeline()


st.sidebar.title("🏥 Readmission Predictor")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate", [
    "📊 Dashboard",
    "📝 Predict from Note",
    "📈 Model Comparison",
    "🔍 Feature Importance",
    "📋 Data Explorer",
])
st.sidebar.markdown("---")
st.sidebar.markdown("**PoC-Grade MedTech Project**")
st.sidebar.markdown("NLP on Clinical Discharge Notes")
st.sidebar.markdown(f"Dataset: {len(df):,} patients")


def risk_gauge(probability):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": "30-Day Readmission Risk"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#1f77b4"},
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


if page == "📊 Dashboard":
    st.title("📊 Patient Readmission Dashboard")
    st.markdown("AI-powered 30-day readmission prediction from clinical discharge notes using NLP.")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Patients", f"{len(df):,}")
    col2.metric("Readmission Rate", f"{df['readmission_30d'].mean():.1%}")
    col3.metric("Features", len(feature_names))
    col4.metric("Best Model", results.iloc[0]["model"].replace("_", " ").title())

    best_roc = results.iloc[0]["roc_auc"]
    st.metric("🏆 Best ROC-AUC", f"{best_roc:.4f}")

    st.markdown("---")
    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("Readmission Distribution")
        counts = df["readmission_label"].value_counts()
        fig = px.bar(x=counts.index, y=counts.values, color=counts.index,
                     color_discrete_map={"NOT READMITTED": "#2ecc71", "READMITTED": "#e74c3c"},
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
    st.subheader("Quick Stats")
    st.dataframe(results[["model", "accuracy", "precision", "recall", "f1", "roc_auc"]].style.format({
        "accuracy": "{:.4f}", "precision": "{:.4f}", "recall": "{:.4f}",
        "f1": "{:.4f}", "roc_auc": "{:.4f}",
    }), use_container_width=True)


elif page == "📝 Predict from Note":
    st.title("📝 Predict Readmission from Clinical Note")
    st.markdown("Paste a patient's discharge summary → get instant readmission risk prediction.")

    # Text area for clinical note input
    default_note = """DISCHARGE SUMMARY

PATIENT: 55555
AGE: 72
SEX: male
LENGTH OF STAY: 8 days

DISCHARGE DIAGNOSIS:
congestive heart failure

HOSPITAL COURSE:
Patient was admitted with congestive heart failure. During hospitalization, patient underwent appropriate workup and treatment. Response to treatment was partial. Patient remains at risk for complications.

DISCHARGE MEDICATIONS:
ejection fraction of 25%, NT-proBNP elevated at 4500, poor medication adherence noted

DISCHARGE INSTRUCTIONS:
Patient was instructed to follow up with primary care physician within 3 days and monitor closely for any warning signs.

FOLLOW-UP:
Cardiology and nephrology follow-up scheduled. Home health services arranged.

SOCIAL HISTORY:
patient has history of multiple readmissions. Patient is a 72-year-old male with history of congestive heart failure.

PHYSICAL EXAM:
Vitals at discharge: BP 160/95, HR 88, RR 22, Temp 37.8C, SpO2 90% on room air.

LABORATORY DATA:
WBC 15, HbA1c 8.5%, Creatinine 2.8, BNP 5200

ASSESSMENT:
Close follow-up and medication reconciliation recommended. Consider case management referral.
"""

    user_note = st.text_area("Paste Discharge Summary:", value=default_note, height=400)

    if st.button("🚀 Predict Readmission Risk", type="primary"):
        with st.spinner("Analyzing clinical note..."):
            # Process the note
            cleaned = clean_clinical_text(user_note)
            tfidf_vec = vectorizer.transform([cleaned])
            clinical_feats = extract_clinical_features(user_note)
            clinical_df = pd.DataFrame([clinical_feats])

            tfidf_df = pd.DataFrame(
                tfidf_vec.toarray(),
                columns=[f"tfidf_{w}" for w in vectorizer.get_feature_names_out()],
            )
            combined = pd.concat([clinical_df.reset_index(drop=True), tfidf_df.reset_index(drop=True)], axis=1)

            # Ensure all features are present (fill missing with 0)
            for f in feature_names:
                if f not in combined.columns:
                    combined[f] = 0
            input_features = combined[feature_names].values

            # Predict
            prediction = best_model.predict(input_features)[0]
            probability = best_model.predict_proba(input_features)[0, 1]

            # Display result
            col_res, col_gauge = st.columns([1, 1])

            with col_res:
                if prediction == 1:
                    st.error("⚠️ HIGH RISK — Readmission Likely")
                    st.markdown(f"**30-day readmission probability: {probability:.1%}**")
                    st.markdown("Recommendations: Close follow-up, medication reconciliation, case management referral.")
                else:
                    st.success("✅ LOW RISK — Readmission Unlikely")
                    st.markdown(f"**30-day readmission probability: {probability:.1%}**")
                    st.markdown("Routine follow-up recommended.")

            with col_gauge:
                st.plotly_chart(risk_gauge(probability), use_container_width=True)

            st.markdown("---")
            st.subheader("🔍 Extracted Clinical Features")
            feats_display = {k: v for k, v in clinical_feats.items()}
            st.dataframe(pd.DataFrame([feats_display]).T.rename(columns={0: "Value"}), use_container_width=True)

            # SHAP explanation (if tree model)
            st.markdown("---")
            st.subheader("🔍 SHAP Explanation — Key Risk Factors")
            try:
                explainer = shap.TreeExplainer(best_model)
                shap_values = explainer.shap_values(input_features)
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]
                sv = shap_values[0] if shap_values.ndim > 1 else shap_values

                contributions = pd.DataFrame({
                    "Feature": feature_names,
                    "SHAP Value": sv,
                    "Impact": ["↑ increases risk" if v > 0 else "↓ decreases risk" for v in sv],
                }).assign(Abs=lambda x: x["SHAP Value"].abs()).sort_values("Abs", ascending=False).head(15)

                st.dataframe(contributions[["Feature", "SHAP Value", "Impact"]].style.format({
                    "SHAP Value": "{:.4f}",
                }), use_container_width=True)

                fig, ax = plt.subplots(figsize=(10, 6))
                shap.waterfall_plot(
                    shap.Explanation(
                        values=sv, base_values=explainer.expected_value,
                        data=input_features[0], feature_names=feature_names,
                    ),
                    max_display=15, show=False,
                )
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close()
            except Exception as e:
                st.info(f"SHAP explanation available for tree-based models. Current model: {type(best_model).__name__}")


elif page == "📈 Model Comparison":
    st.title("📈 Model Comparison")
    st.markdown("Performance across all trained models.")

    st.subheader("Metrics")
    display_df = results[["model", "accuracy", "precision", "recall", "f1", "roc_auc"]].copy()
    display_df["model"] = display_df["model"].str.replace("_", " ").str.title()
    st.dataframe(display_df.style.format({
        "accuracy": "{:.4f}", "precision": "{:.4f}", "recall": "{:.4f}",
        "f1": "{:.4f}", "roc_auc": "{:.4f}",
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


elif page == "🔍 Feature Importance":
    st.title("🔍 Feature Importance")
    st.markdown("Which features drive readmission predictions?")

    # Get feature importance from best model
    if hasattr(best_model, "feature_importances_"):
        importance = best_model.feature_importances_
        imp_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importance,
        }).sort_values("Importance", ascending=False).head(30)

        st.subheader("Top 30 Features")
        fig = px.bar(imp_df.head(20), x="Importance", y="Feature", orientation="h",
                     color="Importance", color_continuous_scale="Viridis")
        fig.update_layout(height=600, yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(imp_df.style.format({"Importance": "{:.4f}"}), use_container_width=True)
    else:
        # Logistic regression — use coefficients
        if hasattr(best_model, "coef_"):
            coef = best_model.coef_[0]
            imp_df = pd.DataFrame({
                "Feature": feature_names,
                "Coefficient": coef,
                "AbsCoefficient": np.abs(coef),
            }).sort_values("AbsCoefficient", ascending=False).head(30)

            st.subheader("Top 30 Features (by coefficient magnitude)")
            fig = px.bar(imp_df.head(20), x="AbsCoefficient", y="Feature", orientation="h",
                         color="AbsCoefficient", color_continuous_scale="Viridis")
            fig.update_layout(height=600, yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig, use_container_width=True)

            st.dataframe(imp_df[["Feature", "Coefficient"]].style.format({"Coefficient": "{:.4f}"}), use_container_width=True)
        else:
            st.warning("Feature importance not available for this model type.")


elif page == "📋 Data Explorer":
    st.title("📋 Clinical Notes Explorer")
    st.markdown("Browse the synthetic clinical discharge notes dataset.")

    # Filter
    filter_label = st.selectbox("Filter by readmission status", ["All", "READMITTED", "NOT READMITTED"])
    if filter_label == "All":
        display_df = df
    else:
        display_df = df[df["readmission_label"] == filter_label]

    st.metric("Patients shown", f"{len(display_df):,}")

    # Show a random note
    if len(display_df) > 0:
        idx = st.slider("Select patient", 0, len(display_df) - 1, 0)
        note = display_df.iloc[idx]

        col1, col2, col3 = st.columns(3)
        col1.metric("Patient ID", note["patient_id"])
        col2.metric("Readmission", note["readmission_label"])
        col3.metric("Note Length", f"{len(note['discharge_note'])} chars")

        st.markdown("---")
        st.subheader("Discharge Summary")
        st.text_area("Clinical Note", value=note["discharge_note"], height=500, key=f"note_{idx}")