from dash import Dash, html, dcc, Input, Output
import pandas as pd
import joblib

from src.train_models import load_models
from src.preprocess_data import preprocess_input
from src.prepare_data import generate_recommendation

app = Dash(__name__)
server = app.server

# Load models
classifier, kmeans, le = load_models()

# Features
features = ["bmi", "Age", "physical_activity_minutes_per_week"]

app.layout = html.Div([
    html.Div([
        html.H1("🩺 Diabetes Risk Decision Support System",
                style={"textAlign": "center", "color": "#2c3e50"}),

        html.P("Enter patient information below to generate risk prediction, clustering and recommendations.",
               style={"textAlign": "center"})
    ]),

    html.Div([
        html.Div([
            html.Label(feature, style={"fontWeight": "bold"}),
            dcc.Input(id=feature, type='number',
                      placeholder=f"Enter {feature}",
                      style={"width": "100%", "padding": "10px", "marginBottom": "10px"})
        ]) for feature in features
    ], style={"width": "40%", "margin": "auto"}),

    html.Div([
        html.Button("Analyze Patient",
                    id="analyze-btn",
                    style={
                        "backgroundColor": "#2ecc71",
                        "color": "white",
                        "padding": "10px 20px",
                        "border": "none",
                        "cursor": "pointer",
                        "marginTop": "20px"
                    })
    ], style={"textAlign": "center"}),

    html.Hr(),

    html.Div(id="risk-output", style={"fontSize": "20px", "marginTop": "20px"}),
    html.Div(id="cluster-output", style={"fontSize": "20px"}),
    html.Div(id="recommendation-output", style={"fontSize": "20px"})
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

            cluster = kmeans.predict(processed)[0]
            
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