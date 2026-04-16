import joblib

def load_models():
    classifier = joblib.load("artifacts/model_classifier.pkl")
    kmeans = joblib.load("artifacts/kmeans_model.pkl")
    le = joblib.load("artifacts/label_encoder.pkl")

    return classifier, kmeans, le