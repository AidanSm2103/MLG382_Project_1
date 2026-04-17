# Dash framework imports for building the web application UI, callbacks, data handling and to load saved ML models and preprocessing objects
from dash import Dash, html, dcc, Input, Output
import pandas as pd
import joblib

# Import custom project modules (backend ML pipeline)
from src.train_models import load_models
from src.preprocess_data import preprocess_input
from src.prepare_data import generate_recommendation

# Create Dash app instance and expose server for deployment
app = Dash(__name__)
server = app.server

# Load trained classification model, clustering model, and label encoder
classifier, kmeans, le = load_models()

# These are the features the user will manually input in the web app
features = ["bmi", "Age", "physical_activity_minutes_per_week"]

# Build web app layout
app.layout = html.Div([
    html.Div([
        # Title section of the dashboard
        html.H1("🩺 Diabetes Risk Decision Support System",
                style={"textAlign": "center", "color": "#2c3e50"}),

        html.P("Enter patient information below to generate risk prediction, clustering and recommendations.",
               style={"textAlign": "center"})
    ]),

    html.Div([
         # Dynamic input fields generated from feature list
        html.Div([
            html.Label(feature, style={"fontWeight": "bold"}),
            dcc.Input(id=feature, type='number',
                      placeholder=f"Enter {feature}",
                      style={"width": "100%", "padding": "10px", "marginBottom": "10px"})
        ]) for feature in features
    ], style={"width": "40%", "margin": "auto"}),

    html.Div([
        # Button to trigger prediction
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

    # Horizontal line separator
    html.Hr(),

    # Output sections
    html.Div(id="risk-output", style={"fontSize": "20px", "marginTop": "20px"}),
    html.Div(id="cluster-output", style={"fontSize": "20px"}),
    html.Div(id="recommendation-output", style={"fontSize": "20px"})
])

@app.callback(
    # Outputs displayed in UI
    [Output("risk-output", "children"),
     Output("cluster-output", "children"),
     Output("recommendation-output", "children")],
    # Trigger event
    Input("analyze-btn", "n_clicks"),

    # User inputs from all feature fields
    [Input(feature, "value") for feature in features]
)

def analyze(n_clicks, *values):
    # Only run when button is clicked
    if n_clicks:

        if not all(v is not None for v in values):
            return "Please fill all inputs", "", ""

        try:
            # Collect user input, align input with full training feature set and load feature columns used during model training
            input_data = dict(zip(features, values))
            feature_cols = joblib.load("artifacts/feature_columns.pkl")
            full_input = {col: 0 for col in feature_cols}
            full_input.update(input_data)
            processed = preprocess_input(full_input)

            # Predict diabetes risk class
            pred = classifier.predict(processed)

            # Convert numeric label back to original category name
            risk_label = le.inverse_transform(pred)[0]

            # Predict segmentation group
            cluster = kmeans.predict(processed)[0]
            
            # Generate recommendation rules
            recommendation = generate_recommendation(input_data)

            return (
                f"Risk Level: {risk_label}",
                f"Patient Segment: Cluster {cluster}",
                f"Recommendation: {recommendation}"
            )

        # Error checking
        except Exception as e:
            return f"Error: {str(e)}", "", ""

    return "", "", ""

# Run application
if __name__ == "__main__":
    app.run(debug=True)