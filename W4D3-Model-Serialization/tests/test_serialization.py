import os
import joblib
import pandas as pd

MODEL_PATH = "artifacts/model_joblib.pkl"
SCALER_PATH = "artifacts/scaler.pkl"

def test_model_file_exists():
    assert os.path.exists(
        MODEL_PATH
    )

def test_scaler_file_exists():
    assert os.path.exists(
        SCALER_PATH
    )

def test_model_loading():
    model = joblib.load(
        MODEL_PATH
    )
    assert model is not None

def test_prediction():
    model = joblib.load(
        MODEL_PATH
    )
    scaler = joblib.load(
        SCALER_PATH
    )
    sample = pd.DataFrame(
        [
            [
                5.1,
                3.5,
                1.4,
                0.2
            ]
        ],
        columns=[
            "sepal length (cm)",
            "sepal width (cm)",
            "petal length (cm)",
            "petal width (cm)"
        ]
    )

    sample_scaled = scaler.transform(
        sample
    )

    prediction = model.predict(
        sample_scaled
    )

    assert len(prediction) == 1