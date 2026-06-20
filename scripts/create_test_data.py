"""Create fallback test data for CI (500K patients)."""
import os
import pandas as pd
import numpy as np

os.makedirs("data", exist_ok=True)

if not os.path.exists("data/diabetes_500k.csv"):
    np.random.seed(42)
    n = 500_000
    df = pd.DataFrame({
        "gender": np.random.choice(["Female", "Male"], n),
        "age": np.clip(np.random.normal(42, 22, n).round(1), 0, 100),
        "hypertension": np.random.binomial(1, 0.075, n),
        "heart_disease": np.random.binomial(1, 0.036, n),
        "smoking_history": np.random.choice(
            ["never", "No Info", "current", "former", "not current", "ever"], n
        ),
        "bmi": np.clip(np.random.normal(27, 6, n).round(1), 10, 60),
        "HbA1c_level": np.clip(np.random.normal(5.5, 1.1, n).round(1), 3, 15),
        "blood_glucose_level": np.clip(
            np.random.normal(138, 40, n).astype(int), 50, 300
        ),
    })
    logits = (
        3.0 * (df["HbA1c_level"] >= 6.5).astype(int)
        + 2.5 * (df["blood_glucose_level"] >= 180).astype(int)
        + 0.5 * (df["age"] > 55).astype(int)
        + 0.3 * (df["bmi"] > 32).astype(int)
        + 0.3 * df["hypertension"]
        + 0.2 * df["heart_disease"]
        - 4.5
    )
    probs = 1 / (1 + np.exp(-logits))
    df["diabetes"] = np.random.binomial(1, probs)
    df.to_csv("data/diabetes_500k.csv", index=False)
    print(f"Created 500K test data: {len(df)} rows, diabetes rate: {df['diabetes'].mean():.2%}")
else:
    print("Data already exists")