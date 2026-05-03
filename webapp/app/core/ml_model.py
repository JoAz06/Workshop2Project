import os
import joblib
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "ml_models")

rf_model = joblib.load(os.path.join(MODEL_DIR, "rf_model.pkl"))
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
ordinalEncoder = joblib.load(os.path.join(MODEL_DIR, "ordinal.pkl"))
X_train = joblib.load(os.path.join(MODEL_DIR, "X_train.pkl"))

numeric_cols = joblib.load(os.path.join(MODEL_DIR, "numeric_cols.pkl"))
categorical_cols = joblib.load(os.path.join(MODEL_DIR, "categorical_cols.pkl"))
cap_cols = joblib.load(os.path.join(MODEL_DIR, "cap_cols.pkl"))
skewed_cols = joblib.load(os.path.join(MODEL_DIR, "skewed_cols.pkl"))
ordinal_cols = joblib.load(os.path.join(MODEL_DIR, "ordinal_cols.pkl"))

def predict_medical_cost(**user_inputs):
    base_row = pd.DataFrame(columns=X_train.columns)
    base_row.loc[0] = 0
    for col in numeric_cols:
        base_row[col] = X_train[col].median()

    for col in categorical_cols:
        base_row[col] = X_train[col].mode()[0]

    for col, val in user_inputs.items():
        if col in base_row.columns:
            base_row[col] = val

    for col in numeric_cols:

        # cap
        if col in cap_cols:
            q1 = X_train[col].quantile(0.25)
            q3 = X_train[col].quantile(0.75)
            iqr = q3 - q1
            base_row[col] = np.clip(base_row[col], q1 - 1.5 * iqr, q3 + 1.5 * iqr)

        # skew
        if col in skewed_cols:
            base_row[col] = np.log1p(base_row[col])

    if len(ordinal_cols) > 0:
        base_row[ordinal_cols] = ordinalEncoder.transform(base_row[ordinal_cols])

    base_row = base_row.reindex(columns=X_train.columns, fill_value=0)

    scaled_row = scaler.transform(base_row)
    pred = rf_model.predict(scaled_row)[0]

    return round(pred, 2)