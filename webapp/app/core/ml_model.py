import os
import joblib
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "ml_models")

xgb_model = joblib.load(os.path.join(MODEL_DIR, "xgb_model.pkl"))
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
ordinalEncoder = joblib.load(os.path.join(MODEL_DIR, "ordinal.pkl"))
X_train = joblib.load(os.path.join(MODEL_DIR, "X_train.pkl"))
X_train_pre_encoding = joblib.load(os.path.join(MODEL_DIR, "X_train_pre_encoding.pkl"))
numeric_cols = joblib.load(os.path.join(MODEL_DIR, "numeric_cols.pkl"))
categorical_cols = joblib.load(os.path.join(MODEL_DIR, "categorical_cols.pkl"))
cap_cols = joblib.load(os.path.join(MODEL_DIR, "cap_cols.pkl"))
skewed_cols = joblib.load(os.path.join(MODEL_DIR, "skewed_cols.pkl"))
ordinal_cols = joblib.load(os.path.join(MODEL_DIR, "ordinal_cols.pkl"))
ohe_cols = joblib.load(os.path.join(MODEL_DIR, "ohe_cols.pkl"))
DecisionTree = joblib.load(os.path.join(MODEL_DIR, "decision_tree.pkl"))
_cap_bounds = joblib.load(os.path.join(MODEL_DIR, "cap_bounds.pkl"))

def predict_medical_cost(**user_inputs):
    base_row = X_train_pre_encoding.iloc[[0]].copy()

    for col in X_train_pre_encoding.columns:
        if col in numeric_cols:
            base_row[col] = X_train_pre_encoding[col].median()
        elif col in categorical_cols:
            base_row[col] = X_train_pre_encoding[col].mode()[0]

    for col, val in user_inputs.items():
        base_row[col] = val

    #Engineering
    base_row['metabolic_risk'] = (
    (base_row['bmi'] > 30).astype(int) +
    (base_row['hba1c'] > 6.5).astype(int) +
    (base_row['ldl'] > 160).astype(int)
    )

    base_row['cardio_risk'] = (
        base_row['hypertension'] +
        base_row['cardiovascular_disease'] +
        (base_row['systolic_bp'] > 140).astype(int)
    )

    base_row['bmi_times_visits'] = (
        base_row['bmi'] * base_row['visits_last_year']
    )

    base_row['total_procedures'] = (
        base_row['proc_imaging_count'] +
        base_row['proc_surgery_count'] +
        base_row['proc_physio_count'] +
        base_row['proc_consult_count'] +
        base_row['proc_lab_count']
    )

    base_row['procedures_per_visit'] = (
        base_row['total_procedures'] / (base_row['visits_last_year'] + 1)
    )

    base_row['avg_cost_per_claim'] = (
        base_row['total_claims_paid'] / (base_row['claims_count'] + 1)
    )

    base_row['avg_days_per_hosp'] = (
        base_row['days_hospitalized_last_3yrs'] /
        (base_row['hospitalizations_last_3yrs'] + 1)
    )

    base_row['dependency_ratio'] = (
        base_row['dependents'] / (base_row['household_size'] + 1)
    )

    base_row['out_of_pocket_index'] = (
        base_row['deductible'] + base_row['copay']
    )

    #Capping
    for col, (lo, hi) in _cap_bounds.items():
        if col in base_row.columns:
            base_row[col] = np.clip(base_row[col], lo, hi)

    #Log transformation
    for col in skewed_cols:
        if col in base_row.columns:
            base_row[col] = np.log1p(base_row[col])

    # Imputation
    for col in numeric_cols:
        if col in base_row.columns and base_row[col].isna().any():
            base_row[col] = X_train_pre_encoding[col].median()

    #Encoding
    base_row[ordinal_cols] = ordinalEncoder.transform(base_row[ordinal_cols])

    present_ohe = [col for col in ohe_cols if col in base_row.columns]
    base_row = pd.get_dummies(base_row, columns=present_ohe, drop_first=True)

    base_row = base_row.reindex(columns=X_train.columns, fill_value=0)

    scaled_row = scaler.transform(base_row)
    pred = xgb_model.predict(scaled_row)[0]

    return round(float(pred), 2)


def predict_if_high_risk(**user_inputs):
    base_row = X_train_pre_encoding.iloc[[0]].copy()

    for col in X_train_pre_encoding.columns:
        if col in numeric_cols:
            base_row[col] = X_train_pre_encoding[col].median()
        elif col in categorical_cols:
            base_row[col] = X_train_pre_encoding[col].mode()[0]

    for col, val in user_inputs.items():
        base_row[col] = val

    #Capping
    for col, (lo, hi) in _cap_bounds.items():
        if col in base_row.columns:
            base_row[col] = np.clip(base_row[col], lo, hi)

    #Log transformation
    for col in skewed_cols:
        if col in base_row.columns:
            base_row[col] = np.log1p(base_row[col])

    # Imputation
    for col in numeric_cols:
        if col in base_row.columns and base_row[col].isna().any():
            base_row[col] = X_train_pre_encoding[col].median()

    #Encoding
    base_row[ordinal_cols] = ordinalEncoder.transform(base_row[ordinal_cols])

    present_ohe = [col for col in ohe_cols if col in base_row.columns]
    base_row = pd.get_dummies(base_row, columns=present_ohe, drop_first=True)

    base_row = base_row.reindex(columns=X_train.columns, fill_value=0)

    scaled_row = scaler.transform(base_row)
    pred = DecisionTree[1].predict(scaled_row)[0]

    return round(float(pred), 2)
