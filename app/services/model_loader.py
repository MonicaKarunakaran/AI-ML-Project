import joblib


MODEL_PATH = "models/iris_model.pkl"
SCALER_PATH = "models/scaler.pkl"

_model = None
_scaler = None


def load_model():
    global _model
    global _scaler

    if _model is None:
        _model = joblib.load(MODEL_PATH)

    if _scaler is None:
        _scaler = joblib.load(SCALER_PATH)

    return _model, _scaler


def get_model():
    return _model


def get_scaler():
    return _scaler