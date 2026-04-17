import pandas as pd
import joblib

def preprocess_input(input_dict):

    feature_cols = joblib.load("artifacts/feature_columns.pkl")
    scaler = joblib.load("artifacts/scaler.pkl")

    # Start with ALL features = 0
    full_input = {col: 0 for col in feature_cols}

    # Map user inputs into correct columns
    for key, value in input_dict.items():
        if key in full_input:
            full_input[key] = value

    # Convert to DataFrame
    df = pd.DataFrame([full_input])

    # Scale
    df_scaled = scaler.transform(df)

    return df_scaled