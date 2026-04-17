from dash import Dash, html, dcc, Input, Output
import pandas as pd

from src.train_models import load_models
from src.preprocess_data import preprocess_input
from src.prepare_data import generate_recommendation

app = Dash(__name__)
server = app.server

# Load models
classifier, kmeans, le = load_models()

# Example features (replace with your actual ones)
features = ["BMI", "Age", "Physical_Activity_Level"]

app.layout = html.Div([
    html.H1("Diabetes Risk Decision Support System"),

    html.H3("Enter Patient Details"),

    html.Div([
        dcc.Input(id=feature, type='number', placeholder=feature)
        for feature in features
    ]),

    html.Button("Analyze Patient", id="analyze-btn"),

    html.H3("Results"),
    html.Div(id="risk-output"),
    html.Div(id="cluster-output"),
    html.Div(id="recommendation-output")
])

@app.callback(
    [Output("risk-output", "children"),
     Output("cluster-output", "children"),
     Output("recommendation-output", "children")],
    Input("analyze-btn", "n_clicks"),
    [Input(feature, "value") for feature in features]
)
def analyze(n_clicks, *values):
    if n_clicks:

        if not all(v is not None for v in values):
            return "Please fill all inputs", "", ""

        try:
            input_data = dict(zip(features, values))

            # Fill ALL missing features with 0
            feature_cols = joblib.load("artifacts/feature_columns.pkl")
            full_input = {col: 0 for col in feature_cols}
            full_input.update(input_data)

            processed = preprocess_input(full_input)

            pred = classifier.predict(processed)
            risk_label = le.inverse_transform(pred)[0]

            #cluster = kmeans.predict(processed)[0]
            cluster = "N/A (segmentation mismatch)"

            recommendation = generate_recommendation(input_data)

            return (
                f"Risk Level: {risk_label}",
                f"Patient Segment: Cluster {cluster}",
                f"Recommendation: {recommendation}"
            )

        except Exception as e:
            return f"Error: {str(e)}", "", ""

    return "", "", ""

if __name__ == "__main__":
    app.run(debug=True)