import os
import joblib
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "ml_models")

Rxgb_model = joblib.load(os.path.join(MODEL_DIR, "Rxgb_model.pkl"))
Rscaler = joblib.load(os.path.join(MODEL_DIR, "Rscaler.pkl"))
RordinalEncoder = joblib.load(os.path.join(MODEL_DIR, "Rordinal.pkl"))
RX_train = joblib.load(os.path.join(MODEL_DIR, "RX_train.pkl"))
RX_train_pre_encoding = joblib.load(os.path.join(MODEL_DIR, "RX_train_pre_encoding.pkl"))
Rnumeric_cols = joblib.load(os.path.join(MODEL_DIR, "Rnumeric_cols.pkl"))
Rcategorical_cols = joblib.load(os.path.join(MODEL_DIR, "Rcategorical_cols.pkl"))
Rcap_cols = joblib.load(os.path.join(MODEL_DIR, "Rcap_cols.pkl"))
Rskewed_cols = joblib.load(os.path.join(MODEL_DIR, "Rskewed_cols.pkl"))
Rordinal_cols = joblib.load(os.path.join(MODEL_DIR, "Rordinal_cols.pkl"))
Rohe_cols = joblib.load(os.path.join(MODEL_DIR, "Rohe_cols.pkl"))
R_cap_bounds = joblib.load(os.path.join(MODEL_DIR, "Rcap_bounds.pkl"))


CDecisionTree = joblib.load(os.path.join(MODEL_DIR, "Cdecision_tree.pkl"))
Cscaler = joblib.load(os.path.join(MODEL_DIR, "Cscaler.pkl"))
CordinalEncoder = joblib.load(os.path.join(MODEL_DIR, "Cordinal.pkl"))
CX_train = joblib.load(os.path.join(MODEL_DIR, "CX_train.pkl"))
CX_train_pre_encoding = joblib.load(os.path.join(MODEL_DIR, "CX_train_pre_encoding.pkl"))
Cnumeric_cols = joblib.load(os.path.join(MODEL_DIR, "Cnumeric_cols.pkl"))
Ccategorical_cols = joblib.load(os.path.join(MODEL_DIR, "Ccategorical_cols.pkl"))
Ccap_cols = joblib.load(os.path.join(MODEL_DIR, "Ccap_cols.pkl"))
Cskewed_cols = joblib.load(os.path.join(MODEL_DIR, "Cskewed_cols.pkl"))
Cordinal_cols = joblib.load(os.path.join(MODEL_DIR, "Cordinal_cols.pkl"))
Cohe_cols = joblib.load(os.path.join(MODEL_DIR, "Cohe_cols.pkl"))
C_cap_bounds = joblib.load(os.path.join(MODEL_DIR, "Ccap_bounds.pkl"))

def predict_medical_cost(**user_inputs):
    base_row = RX_train_pre_encoding.iloc[[0]].copy()

    for col in RX_train_pre_encoding.columns:
        if col in Rnumeric_cols:
            base_row[col] = RX_train_pre_encoding[col].median()
        elif col in Rcategorical_cols:
            base_row[col] = RX_train_pre_encoding[col].mode()[0]

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
    for col, (lo, hi) in R_cap_bounds.items():
        if col in base_row.columns:
            base_row[col] = np.clip(base_row[col], lo, hi)

    #Log transformation
    for col in Rskewed_cols:
        if col in base_row.columns:
            base_row[col] = np.log1p(base_row[col])

    # Imputation
    for col in Rnumeric_cols:
        if col in base_row.columns and base_row[col].isna().any():
            base_row[col] = RX_train_pre_encoding[col].median()

    #Encoding
    base_row[Rordinal_cols] = RordinalEncoder.transform(base_row[Rordinal_cols])

    present_ohe = [col for col in Rohe_cols if col in base_row.columns]
    base_row = pd.get_dummies(base_row, columns=present_ohe, drop_first=True)

    base_row = base_row.reindex(columns=RX_train.columns, fill_value=0)

    scaled_row = Rscaler.transform(base_row)
    pred = Rxgb_model.predict(scaled_row)[0]

    return round(float(pred), 2)


def predict_if_high_risk(**user_inputs):
    base_row = CX_train_pre_encoding.iloc[[0]].copy()

    for col in CX_train_pre_encoding.columns:
        if col in Cnumeric_cols:
            base_row[col] = CX_train_pre_encoding[col].median()
        elif col in Ccategorical_cols:
            base_row[col] = CX_train_pre_encoding[col].mode()[0]

    for col, val in user_inputs.items():
        base_row[col] = val

    #Capping
    for col, (lo, hi) in C_cap_bounds.items():
        if col in base_row.columns:
            base_row[col] = np.clip(base_row[col], lo, hi)

    #Log transformation
    for col in Cskewed_cols:
        if col in base_row.columns:
            base_row[col] = np.log1p(base_row[col])

    # Imputation
    for col in Cnumeric_cols:
        if col in base_row.columns and base_row[col].isna().any():
            base_row[col] = CX_train_pre_encoding[col].median()

    #Encoding
    base_row[Cordinal_cols] = CordinalEncoder.transform(base_row[Cordinal_cols])

    present_ohe = [col for col in Cohe_cols if col in base_row.columns]
    base_row = pd.get_dummies(base_row, columns=present_ohe, drop_first=True)

    base_row = base_row.reindex(columns=CX_train.columns, fill_value=0)

    scaled_row = Cscaler.transform(base_row)
    pred = CDecisionTree[1].predict(scaled_row)[0]

    return round(float(pred), 2)
