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
        try:
            input_data = dict(zip(features, values))

            processed = preprocess_input(input_data)

            # Prediction
            pred = classifier.predict(processed)
            risk_label = le.inverse_transform(pred)[0]

            # Cluster
            cluster = kmeans.predict(processed)[0]

            # Recommendation
            recommendation = generate_recommendation(input_data)

            return (
                f"Risk Level: {risk_label}",
                f"Patient Segment: Cluster {cluster}",
                f"Recommendation: {recommendation}"
            )

        except:
            return "Error in input", "", ""

    return "", "", ""

if __name__ == "__main__":
    app.run(debug=True)