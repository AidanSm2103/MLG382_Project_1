import joblib

# Loads trained ML models from disk
def load_models():
    # Classification model (predicts diabetes risk stage)
    classifier = joblib.load("artifacts/model_classifier.pkl")

    # KMeans clustering model (patient segmentation)
    kmeans = joblib.load("artifacts/kmeans_model.pkl")

    # Label encoder (converts numeric predictions back to labels)
    le = joblib.load("artifacts/label_encoder.pkl")

    return classifier, kmeans, le