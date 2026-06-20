"""
Streamlit Dashboard — MedTech AI Suite
Page 1-5: Readmission Predictor (NLP on clinical notes)
Page 6-8: Diabetes Predictor (real clinical data, 100K patients)
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

from nlp_pipeline import clean_clinical_text, extract_clinical_features, prepare_data
from model import train_and_compare as train_readmission
from diabetes_model import load_diabetes_data, preprocess as preprocess_diabetes, train_and_compare as train_diabetes

MODELS_DIR = Path(PROJECT_ROOT) / "models"

st.set_page_config(page_title="MedTech AI Suite", page_icon="🏥", layout="wide")


@st.cache_data
def load_readmission_data():
    return pd.read_csv(PROJECT_ROOT / "data" / "clinical_notes.csv")


@st.cache_data
def load_diabetes_data_cached():
    return load_diabetes_data()


@st.cache_resource
def get_readmission_pipeline():
    df = load_readmission_data()
    X_train, X_test, y_train, y_test, features, vec = prepare_data(df)
    results, models = train_readmission(X_train, X_test, y_train, y_test)
    best = joblib.load(MODELS_DIR / "best_model.joblib")
    return df, X_train, X_test, y_train, y_test, features, vec, results, best


@st.cache_resource
def get_diabetes_pipeline():
    df = load_diabetes_data_cached()
    X_train, X_test, y_train, y_test, features, prep = preprocess_diabetes(df)
    results, models = train_diabetes(X_train, X_test, y_train, y_test)
    best = joblib.load(MODELS_DIR / "diabetes_best_model.joblib")
    return df, X_train, X_test, y_train, y_test, features, prep, results, best


with st.spinner("Loading models..."):
    ra_df, ra_Xtr, ra_Xte, ra_ytr, ra_yte, ra_feat, ra_vec, ra_results, ra_best = get_readmission_pipeline()

with st.spinner("Loading diabetes model (100K patients)..."):
    db_df, db_Xtr, db_Xte, db_ytr, db_yte, db_feat, db_prep, db_results, db_best = get_diabetes_pipeline()


st.sidebar.title("🏥 MedTech AI Suite")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate", [
    "📊 Overview",
    "📝 Readmission Predictor",
    "🩺 Diabetes Predictor",
    "📈 Model Comparison",
    "🔍 Feature Importance",
    "📋 Data Explorer",
])
st.sidebar.markdown("---")
st.sidebar.markdown("**MedTech AI Suite**")
st.sidebar.markdown(f"Readmission: {len(ra_df):,} notes")
st.sidebar.markdown(f"Diabetes: {len(db_df):,} patients")


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


if page == "📊 Overview":
    st.title("🏥 MedTech AI Suite — Overview")
    st.markdown("Two AI models for healthcare: **Readmission Prediction** (NLP) + **Diabetes Prediction** (clinical data)")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📝 Readmission Predictor (NLP)")
        st.metric("Patients", f"{len(ra_df):,}")
        st.metric("Readmission Rate", f"{ra_df['readmission_30d'].mean():.1%}")
        st.metric("Features", len(ra_feat))
        st.metric("Best ROC-AUC", f"{ra_results.iloc[0]['roc_auc']:.4f}")

    with col2:
        st.subheader("🩺 Diabetes Predictor (Real Data)")
        st.metric("Patients", f"{len(db_df):,}")
        st.metric("Diabetes Rate", f"{db_df['diabetes'].mean():.1%}")
        st.metric("Features", len(db_feat))
        st.metric("Best ROC-AUC", f"{db_results.iloc[0]['roc_auc']:.4f}")

    st.markdown("---")
    st.subheader("Model Performance Comparison")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Readmission (NLP)**")
        st.dataframe(ra_results[["model", "roc_auc", "f1", "accuracy"]].style.format({
            "roc_auc": "{:.4f}", "f1": "{:.4f}", "accuracy": "{:.4f}"
        }), use_container_width=True)

    with col2:
        st.markdown("**Diabetes (Clinical)**")
        st.dataframe(db_results[["model", "roc_auc", "f1", "accuracy"]].style.format({
            "roc_auc": "{:.4f}", "f1": "{:.4f}", "accuracy": "{:.4f}"
        }), use_container_width=True)


elif page == "📝 Readmission Predictor":
    st.title("📝 Readmission Prediction from Clinical Notes")
    st.markdown("Paste a discharge summary → get 30-day readmission risk prediction.")

    default_note = """DISCHARGE SUMMARY

PATIENT: 55555
AGE: 72
SEX: male
LENGTH OF STAY: 8 days

DISCHARGE DIAGNOSIS:
congestive heart failure

HOSPITAL COURSE:
Patient was admitted with congestive heart failure. Response to treatment was partial. Patient remains at risk for complications.

DISCHARGE MEDICATIONS:
ejection fraction of 25%, NT-proBNP elevated at 4500, poor medication adherence noted

FOLLOW-UP:
Cardiology and nephrology follow-up scheduled. Home health services arranged.

SOCIAL HISTORY:
patient has history of multiple readmissions. Patient is a 72-year-old male.

PHYSICAL EXAM:
Vitals at discharge: BP 160/95, HR 88, RR 22, Temp 37.8C, SpO2 90% on room air.

LABORATORY DATA:
WBC 15, HbA1c 8.5%, Creatinine 2.8, BNP 5200

ASSESSMENT:
Close follow-up and medication reconciliation recommended. Consider case management referral.
"""
    user_note = st.text_area("Paste Discharge Summary:", value=default_note, height=350)

    if st.button("🚀 Predict Readmission Risk", type="primary"):
        with st.spinner("Analyzing clinical note..."):
            cleaned = clean_clinical_text(user_note)
            tfidf_vec = ra_vec.transform([cleaned])
            clinical_feats = extract_clinical_features(user_note)
            clinical_df = pd.DataFrame([clinical_feats])
            tfidf_df = pd.DataFrame(tfidf_vec.toarray(), columns=[f"tfidf_{w}" for w in ra_vec.get_feature_names_out()])
            combined = pd.concat([clinical_df.reset_index(drop=True), tfidf_df.reset_index(drop=True)], axis=1)
            for f in ra_feat:
                if f not in combined.columns:
                    combined[f] = 0
            input_features = combined[ra_feat].values

            prediction = ra_best.predict(input_features)[0]
            probability = ra_best.predict_proba(input_features)[0, 1]

            col_res, col_gauge = st.columns([1, 1])
            with col_res:
                if prediction == 1:
                    st.error("⚠️ HIGH RISK — Readmission Likely")
                    st.markdown(f"**30-day readmission probability: {probability:.1%}**")
                else:
                    st.success("✅ LOW RISK — Readmission Unlikely")
                    st.markdown(f"**30-day readmission probability: {probability:.1%}**")
            with col_gauge:
                st.plotly_chart(risk_gauge(probability, "30-Day Readmission Risk"), use_container_width=True)

            st.subheader("🔍 Extracted Clinical Features")
            st.dataframe(pd.DataFrame([clinical_feats]).T.rename(columns={0: "Value"}), use_container_width=True)


elif page == "🩺 Diabetes Predictor":
    st.title("🩺 Diabetes Risk Predictor")
    st.markdown("Enter patient vitals → get diabetes risk prediction. **Model trained on 100,000 real patients.**")
    st.metric("Model ROC-AUC", f"{db_results.iloc[0]['roc_auc']:.4f}")

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
        with st.spinner("Running prediction..."):
            # Build input
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

            # Transform through preprocessor
            input_processed = db_prep.transform(input_df)
            input_processed = np.nan_to_num(input_processed.astype(np.float32), nan=-999)

            prediction = db_best.predict(input_processed)[0]
            probability = db_best.predict_proba(input_processed)[0, 1]

            col_res, col_gauge = st.columns([1, 1])
            with col_res:
                if prediction == 1:
                    st.error("⚠️ HIGH RISK — Diabetes Likely")
                    st.markdown(f"**Diabetes probability: {probability:.1%}**")
                    st.markdown("Recommend: Confirm with fasting glucose test + HbA1c. Refer to endocrinologist.")
                else:
                    st.success("✅ LOW RISK — Diabetes Unlikely")
                    st.markdown(f"**Diabetes probability: {probability:.1%}**")
                    st.markdown("Routine screening recommended at next checkup.")
            with col_gauge:
                st.plotly_chart(risk_gauge(probability, "Diabetes Risk Score"), use_container_width=True)

            # SHAP explanation
            st.markdown("---")
            st.subheader("🔍 SHAP Explanation — Key Risk Factors")
            try:
                explainer = shap.TreeExplainer(db_best)
                shap_values = explainer.shap_values(input_processed)
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]
                sv = shap_values[0] if shap_values.ndim > 1 else shap_values
                base = explainer.expected_value
                if isinstance(base, (list, np.ndarray)):
                    base = base[1]

                fig, ax = plt.subplots(figsize=(10, 5))
                shap.waterfall_plot(
                    shap.Explanation(values=sv, base_values=base,
                                     data=input_processed[0], feature_names=db_feat),
                    max_display=8, show=False,
                )
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close()

                contributions = pd.DataFrame({
                    "Feature": db_feat,
                    "SHAP Value": sv,
                    "Impact": ["↑ increases risk" if v > 0 else "↓ decreases risk" for v in sv],
                }).assign(Abs=lambda x: x["SHAP Value"].abs()).sort_values("Abs", ascending=False).head(8)
                st.dataframe(contributions[["Feature", "SHAP Value", "Impact"]].style.format({
                    "SHAP Value": "{:.4f}"
                }), use_container_width=True)
            except Exception as e:
                st.info(f"SHAP explanation: {e}")


elif page == "📈 Model Comparison":
    st.title("📈 Model Comparison")

    st.subheader("📝 Readmission Predictor (NLP)")
    st.dataframe(ra_results[["model", "accuracy", "precision", "recall", "f1", "roc_auc"]].style.format({
        "accuracy": "{:.4f}", "precision": "{:.4f}", "recall": "{:.4f}",
        "f1": "{:.4f}", "roc_auc": "{:.4f}",
    }), use_container_width=True)

    st.subheader("🩺 Diabetes Predictor (Real Data)")
    st.dataframe(db_results[["model", "accuracy", "precision", "recall", "f1", "roc_auc"]].style.format({
        "accuracy": "{:.4f}", "precision": "{:.4f}", "recall": "{:.4f}",
        "f1": "{:.4f}", "roc_auc": "{:.4f}",
    }), use_container_width=True)

    # ROC-AUC comparison chart
    st.markdown("---")
    st.subheader("ROC-AUC Comparison")
    combined_results = pd.concat([
        ra_results.assign(project="Readmission (NLP)"),
        db_results.assign(project="Diabetes (Clinical)"),
    ])
    fig = px.bar(combined_results, x="model", y="roc_auc", color="project",
                 barmode="group", range_y=[0.5, 1.0],
                 labels={"model": "Model", "roc_auc": "ROC-AUC", "project": "Project"})
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)


elif page == "🔍 Feature Importance":
    st.title("🔍 Feature Importance")

    project = st.selectbox("Select project", ["Diabetes Predictor", "Readmission Predictor"])

    if project == "Diabetes Predictor":
        if hasattr(db_best, "feature_importances_"):
            imp = db_best.feature_importances_
            imp_df = pd.DataFrame({"Feature": db_feat, "Importance": imp}).sort_values("Importance", ascending=False)
            fig = px.bar(imp_df, x="Importance", y="Feature", orientation="h",
                         color="Importance", color_continuous_scale="Viridis")
            fig.update_layout(height=400, yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(imp_df.style.format({"Importance": "{:.4f}"}), use_container_width=True)
        else:
            st.info("Feature importance available for tree models.")

    else:
        if hasattr(ra_best, "coef_"):
            coef = ra_best.coef_[0]
            imp_df = pd.DataFrame({"Feature": ra_feat, "Coefficient": coef}).assign(Abs=lambda x: x.Coefficient.abs()).sort_values("Abs", ascending=False).head(20)
            fig = px.bar(imp_df, x="Abs", y="Feature", orientation="h", color="Abs", color_continuous_scale="Viridis")
            fig.update_layout(height=500, yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig, use_container_width=True)
        elif hasattr(ra_best, "feature_importances_"):
            imp = ra_best.feature_importances_
            imp_df = pd.DataFrame({"Feature": ra_feat, "Importance": imp}).sort_values("Importance", ascending=False).head(20)
            fig = px.bar(imp_df, x="Importance", y="Feature", orientation="h", color="Importance", color_continuous_scale="Viridis")
            fig.update_layout(height=500, yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig, use_container_width=True)


elif page == "📋 Data Explorer":
    st.title("📋 Data Explorer")

    dataset = st.selectbox("Select dataset", ["Diabetes (100K real patients)", "Readmission (5K synthetic notes)"])

    if "Diabetes" in dataset:
        st.metric("Total patients", f"{len(db_df):,}")
        st.metric("Diabetes rate", f"{db_df['diabetes'].mean():.1%}")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Age Distribution by Diabetes Status")
            fig = px.histogram(db_df, x="age", color="diabetes", nbins=50,
                               color_discrete_map={0: "#2ecc71", 1: "#e74c3c"},
                               labels={"diabetes": "Diabetes"})
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("HbA1c vs Blood Glucose")
            sample = db_df.sample(min(5000, len(db_df)))
            fig = px.scatter(sample, x="HbA1c_level", y="blood_glucose_level", color="diabetes",
                             color_discrete_map={0: "#2ecc71", 1: "#e74c3c"},
                             labels={"diabetes": "Diabetes"}, opacity=0.5)
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)

        st.dataframe(db_df.head(20), use_container_width=True)

    else:
        st.metric("Total notes", f"{len(ra_df):,}")
        filter_label = st.selectbox("Filter", ["All", "READMITTED", "NOT READMITTED"])
        display = ra_df if filter_label == "All" else ra_df[ra_df["readmission_label"] == filter_label]

        idx = st.slider("Select patient", 0, max(0, len(display) - 1), 0)
        note = display.iloc[idx]
        col1, col2 = st.columns(3)
        col1.metric("Patient ID", note["patient_id"])
        col2.metric("Readmission", note["readmission_label"])
        col3.metric("Note Length", f"{len(note['discharge_note'])} chars")

        st.text_area("Clinical Note", value=note["discharge_note"], height=400, key=f"explorer_{idx}")