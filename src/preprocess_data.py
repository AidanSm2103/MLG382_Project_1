import pandas as pd
import joblib

def preprocess_input(input_dict):
    feature_cols = joblib.load("artifacts/feature_columns.pkl")

    df = pd.DataFrame([input_dict])

    df = pd.get_dummies(df)

    # align columns
    for col in feature_cols:
        if col not in df:
            df[col] = 0

    df = df[feature_cols]

    scaler = joblib.load("artifacts/scaler.pkl")
    df_scaled = scaler.transform(df)

    return df_scaled