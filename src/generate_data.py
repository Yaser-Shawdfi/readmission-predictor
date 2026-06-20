"""
Generate a realistic synthetic clinical notes dataset for readmission prediction.
Creates 5,000 synthetic patient discharge summaries with readmission labels.
"""

import random

import numpy as np
import pandas as pd

random.seed(42)
np.random.seed(42)

HIGH_RISK_CONDITIONS = [
    "congestive heart failure",
    "chronic obstructive pulmonary disease",
    "diabetic ketoacidosis",
    "acute myocardial infarction",
    "pneumonia",
    "chronic kidney disease",
    "cirrhosis",
    "sepsis",
]

LOW_RISK_CONDITIONS = [
    "routine post-operative recovery",
    "elective surgery follow-up",
    "minor fracture",
    "dehydration",
    "mild hypertension",
    "healthy pregnancy delivery",
    "routine checkup",
    "skin infection",
]

HIGH_RISK_FACTORS = [
    "patient has history of multiple readmissions",
    "poor medication adherence noted",
    "lives alone with limited social support",
    "discharged against medical advice",
    "uncontrolled blood sugar levels at discharge",
    "elevated creatinine levels persisting",
    "oxygen saturation below 92% on room air",
    "positive blood cultures on day of discharge",
    "BMI greater than 35",
    "smoker with 40 pack-year history",
    "HbA1c of 9.8%",
    "ejection fraction of 25%",
    "NT-proBNP elevated at 4500",
    "severe peripheral edema",
    "inability to perform ADLs independently",
]

LOW_RISK_FACTORS = [
    "stable vitals at discharge",
    "excellent medication compliance expected",
    "strong family support system",
    "patient educated on discharge instructions",
    "all labs within normal limits",
    "ambulating independently",
    "pain well controlled on oral medications",
    "follow-up appointment scheduled within 3 days",
    "patient motivated and engaged in care plan",
    "weight stable and no fluid overload",
]

SECTIONS = [
    "DISCHARGE DIAGNOSIS:",
    "HOSPITAL COURSE:",
    "DISCHARGE MEDICATIONS:",
    "DISCHARGE INSTRUCTIONS:",
    "FOLLOW-UP:",
    "SOCIAL HISTORY:",
    "PAST MEDICAL HISTORY:",
    "PHYSICAL EXAM:",
    "LABORATORY DATA:",
    "ASSESSMENT:",
]


def _pick(lst, n=3):
    return ", ".join(random.sample(lst, min(n, len(lst))))


def generate_clinical_note(high_risk: bool) -> str:
    """Generate a realistic discharge summary with some noise for realism."""
    conditions = HIGH_RISK_CONDITIONS if high_risk else LOW_RISK_CONDITIONS
    factors = HIGH_RISK_FACTORS if high_risk else LOW_RISK_FACTORS
    other_factors = LOW_RISK_FACTORS if high_risk else HIGH_RISK_FACTORS

    # Add 10% noise: sometimes include a factor from the other class
    selected_factors = list(random.sample(factors, min(3, len(factors))))
    if random.random() < 0.15:
        selected_factors.append(random.choice(other_factors))

    diagnosis = random.choice(conditions)
    # 10% chance of cross-contamination in diagnosis
    if random.random() < 0.08:
        other_conditions = LOW_RISK_CONDITIONS if high_risk else HIGH_RISK_CONDITIONS
        diagnosis = random.choice(other_conditions)

    age = random.randint(45, 90) if high_risk else random.randint(25, 75)
    # Add age noise
    if random.random() < 0.15:
        age = random.randint(30, 85)
    sex = random.choice(["male", "female"])
    los = random.randint(5, 14) if high_risk else random.randint(1, 4)
    # Add LOS noise
    if random.random() < 0.15:
        los = random.randint(2, 10)

    if high_risk:
        course = "Response to treatment was partial. Patient remains at risk for complications."
        labs = f"WBC {random.randint(12, 20)}, HbA1c {random.uniform(7, 11):.1f}%, Creatinine {random.uniform(1.5, 3.5):.1f}, BNP {random.randint(2000, 6000)}"
        assessment = "Close follow-up and medication reconciliation recommended. Consider case management referral."
        fu = "Cardiology and nephrology follow-up scheduled. Home health services arranged. Patient counseled on medication adherence and dietary restrictions."
    else:
        course = "Patient responded well to treatment with significant improvement."
        labs = f"WBC {random.randint(4, 11)}, HbA1c {random.uniform(5, 6.5):.1f}%, Creatinine {random.uniform(0.6, 1.2):.1f}"
        assessment = "Routine follow-up recommended. Patient stable for discharge."
        fu = "Routine follow-up with PCP. No specialty follow-up required at this time."

    bp_sys = random.randint(110, 180)
    bp_dia = random.randint(60, 110)
    hr = random.randint(60, 110)
    rr = random.randint(16, 28)
    temp = random.uniform(36.5, 38.5)
    spo2 = random.randint(88, 99)

    note = f"""DISCHARGE SUMMARY

PATIENT: {random.randint(10000, 99999)}
AGE: {age}
SEX: {sex}
LENGTH OF STAY: {los} days

{SECTIONS[0]}
{diagnosis}

{SECTIONS[1]}
Patient was admitted with {diagnosis}. During hospitalization, patient underwent appropriate workup and treatment. {course}

{SECTIONS[2]}
{", ".join(selected_factors[:3])}

{SECTIONS[3]}
Patient was instructed to follow up with primary care physician within 3 days and monitor closely for any warning signs.

{SECTIONS[4]}
{fu}

{SECTIONS[5]}
{selected_factors[0] if selected_factors else ""}. Patient is a {age}-year-old {sex} with history of {diagnosis}.

{SECTIONS[6]}
{selected_factors[1] if len(selected_factors) > 1 else selected_factors[0] if selected_factors else ""}.

{SECTIONS[7]}
Vitals at discharge: BP {bp_sys}/{bp_dia}, HR {hr}, RR {rr}, Temp {temp:.1f}C, SpO2 {spo2}% on room air.

{SECTIONS[8]}
{labs}

{SECTIONS[9]}
{assessment}
"""
    return note.strip()


def generate_dataset(n_samples: int = 5000) -> pd.DataFrame:
    """Generate full dataset with ~30% positive (readmitted) class."""
    notes = []
    labels = []

    n_high = int(n_samples * 0.30)
    n_low = n_samples - n_high

    for _ in range(n_high):
        notes.append(generate_clinical_note(high_risk=True))
        labels.append(1)

    for _ in range(n_low):
        notes.append(generate_clinical_note(high_risk=False))
        labels.append(0)

    combined = list(zip(notes, labels))
    random.shuffle(combined)
    notes, labels = zip(*combined)

    return pd.DataFrame(
        {
            "patient_id": [f"P{10000 + i}" for i in range(len(notes))],
            "discharge_note": notes,
            "readmission_30d": labels,
            "readmission_label": ["READMITTED" if l == 1 else "NOT READMITTED" for l in labels],
        }
    )


if __name__ == "__main__":
    from pathlib import Path

    data_dir = Path(__file__).resolve().parent.parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    df = generate_dataset(n_samples=5000)
    filepath = data_dir / "clinical_notes.csv"
    df.to_csv(filepath, index=False)

    print(f"Generated {len(df)} clinical notes")
    print(f"  Readmitted: {df['readmission_30d'].sum()} ({df['readmission_30d'].mean():.1%})")
    print(f"  Not readmitted: {(1 - df['readmission_30d']).sum()}")
    print(f"  Saved to: {filepath}")
    print("\nSample (readmitted):")
    print(df[df["readmission_30d"] == 1].iloc[0]["discharge_note"][:400])
