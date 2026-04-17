import pandas as pd
import joblib

# Converts raw user input into model-ready format
def preprocess_input(input_dict):

    # Load training feature order
    feature_cols = joblib.load("artifacts/feature_columns.pkl")

    # Load scaler used during training
    scaler = joblib.load("artifacts/scaler.pkl")

    # Start with all features = 0
    full_input = {col: 0 for col in feature_cols}

    # Replace matching keys with actual user input values
    for key, value in input_dict.items():
        if key in full_input:
            full_input[key] = value

    # Convert to DataFrame
    df = pd.DataFrame([full_input])

    # Scale transformation
    df_scaled = scaler.transform(df)

    return df_scaled