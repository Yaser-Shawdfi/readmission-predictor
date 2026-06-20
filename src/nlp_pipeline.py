"""
NLP preprocessing pipeline for clinical notes.
TF-IDF vectorization with clinical text cleaning + feature extraction.
"""

import logging
import re
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

logger = logging.getLogger("readmission.nlp")

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def clean_clinical_text(text: str) -> str:
    """Clean clinical note text for NLP processing."""
    # Convert to lowercase
    text = text.lower()
    # Remove patient ID numbers
    text = re.sub(r"patient:\s*\d+", "", text)
    # Remove standalone numbers (keep words with numbers like hba1c)
    text = re.sub(r"\b\d+\b", "", text)
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)
    # Remove section headers (keep the content)
    for header in [
        "discharge summary",
        "discharge diagnosis:",
        "hospital course:",
        "discharge medications:",
        "discharge instructions:",
        "follow-up:",
        "social history:",
        "past medical history:",
        "physical exam:",
        "laboratory data:",
        "assessment:",
    ]:
        text = text.replace(header, "")
    return text.strip()


def extract_clinical_features(text: str) -> dict:
    """Extract structured features from clinical notes."""
    text_lower = text.lower()

    features = {}

    # Risk keywords
    risk_keywords = [
        "readmission",
        "poor medication adherence",
        "lives alone",
        "against medical advice",
        "uncontrolled",
        "elevated creatinine",
        "oxygen saturation below",
        "positive blood cultures",
        "ejection fraction",
        "nt-probnp",
        "peripheral edema",
        "inability to perform",
        "partial",
        "remains at risk",
        "case management",
    ]
    features["risk_keyword_count"] = sum(1 for kw in risk_keywords if kw in text_lower)

    # Protective keywords
    protective_keywords = [
        "stable vitals",
        "excellent medication",
        "family support",
        "normal limits",
        "ambulating independently",
        "well controlled",
        "low risk",
        "routine follow-up",
        "motivated",
    ]
    features["protective_keyword_count"] = sum(1 for kw in protective_keywords if kw in text_lower)

    # High-risk conditions
    high_risk_conditions = [
        "heart failure",
        "copd",
        "diabetic ketoacidosis",
        "myocardial infarction",
        "pneumonia",
        "kidney disease",
        "cirrhosis",
        "sepsis",
    ]
    features["high_risk_condition_count"] = sum(1 for c in high_risk_conditions if c in text_lower)

    # Numeric indicators
    features["mentions_hba1c"] = int("hba1c" in text_lower)
    features["mentions_creatinine"] = int("creatinine" in text_lower)
    features["mentions_bnp"] = int("bnp" in text_lower or "nt-probnp" in text_lower)
    features["mentions_ef"] = int("ejection fraction" in text_lower)
    features["mentions_wbc"] = int("wbc" in text_lower)

    # Length of stay (extract from text)
    los_match = re.search(r"length of stay:\s*(\d+)\s*days", text_lower)
    features["length_of_stay"] = int(los_match.group(1)) if los_match else 0

    # Age
    age_match = re.search(r"age:\s*(\d+)", text_lower)
    features["age"] = int(age_match.group(1)) if age_match else 0

    # Note length
    features["note_length"] = len(text.split())

    return features


def prepare_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Full NLP pipeline: clean text → TF-IDF + clinical features → train/test split.
    Returns X_train, X_test, y_train, y_test, feature_names, vectorizer.
    """
    logger.info(f"Preparing data: {len(df)} notes")

    # Clean text
    cleaned = df["discharge_note"].apply(clean_clinical_text)

    # TF-IDF vectorization
    vectorizer = TfidfVectorizer(
        max_features=500,
        ngram_range=(1, 2),
        stop_words="english",
        min_df=5,
        max_df=0.95,
        sublinear_tf=True,
    )
    tfidf_matrix = vectorizer.fit_transform(cleaned)

    # Extract clinical features
    clinical_feats = df["discharge_note"].apply(extract_clinical_features)
    clinical_df = pd.DataFrame(list(clinical_feats))

    # Combine TF-IDF + clinical features
    tfidf_df = pd.DataFrame(
        tfidf_matrix.toarray(), columns=[f"tfidf_{w}" for w in vectorizer.get_feature_names_out()]
    )
    combined = pd.concat(
        [clinical_df.reset_index(drop=True), tfidf_df.reset_index(drop=True)], axis=1
    )

    feature_names = list(combined.columns)
    X = combined.values
    y = df["readmission_30d"].values

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    logger.info(f"Train: {X_train.shape}, Test: {X_test.shape}")
    logger.info(
        f"Features: {len(feature_names)} ({len(clinical_df.columns)} clinical + {len(tfidf_df.columns)} TF-IDF)"
    )

    # Save vectorizer
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, MODELS_DIR / "tfidf_vectorizer.joblib")

    return X_train, X_test, y_train, y_test, feature_names, vectorizer


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    from pathlib import Path

    data_path = Path(__file__).resolve().parent.parent / "data" / "clinical_notes.csv"
    df = pd.read_csv(data_path)

    X_train, X_test, y_train, y_test, features, vec = prepare_data(df)
    print(f"\nTrain: {X_train.shape}, Test: {X_test.shape}")
    print(f"Train readmission rate: {y_train.mean():.1%}")
    print(f"Test readmission rate: {y_test.mean():.1%}")
    print(f"\nTop TF-IDF features: {vec.get_feature_names_out()[:20]}")
