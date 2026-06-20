"""Create fallback test data for CI."""
import os
import pandas as pd
import numpy as np

os.makedirs("data", exist_ok=True)

if not os.path.exists("data/diabetes_100k.csv"):
    np.random.seed(42)
    n = 500
    df = pd.DataFrame({
        "gender": np.random.choice(["M", "F"], n),
        "age": np.random.randint(20, 80, n),
        "hypertension": np.random.randint(0, 2, n),
        "heart_disease": np.random.randint(0, 2, n),
        "smoking_history": np.random.choice(["never", "current"], n),
        "bmi": np.random.uniform(15, 40, n),
        "HbA1c_level": np.random.uniform(3, 10, n),
        "blood_glucose_level": np.random.randint(70, 200, n),
        "diabetes": np.random.randint(0, 2, n),
    })
    df.to_csv("data/diabetes_100k.csv", index=False)
    print(f"Created test data: {len(df)} rows")