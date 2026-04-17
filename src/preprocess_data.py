import pandas as pd
import joblib

def preprocess_input(input_dict):
    feature_cols = joblib.load("artifacts/feature_columns.pkl")
    scaler = joblib.load("artifacts/scaler.pkl")

    # Convert to DataFrame
    df = pd.DataFrame([input_dict])

    # One-hot encode
    df = pd.get_dummies(df)

    # Add missing columns
    for col in feature_cols:
        if col not in df:
            df[col] = 0

    # Remove extra columns
    df = df[feature_cols]

    # Scale
    df_scaled = scaler.transform(df)

    return df_scaled