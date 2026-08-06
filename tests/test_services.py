import pandas as pd

from app.services.model_loader import load_model
from app.services.predictor import predict


def test_load_model():

    model, scaler = load_model()

    assert model is not None

    assert scaler is not None


def test_predict():

    sample = pd.DataFrame(
        [[5.1,3.5,1.4,0.2]],
        columns=[
            "sepal_length",
            "sepal_width",
            "petal_length",
            "petal_width"
        ]
    )

    result = predict(sample)

    assert "prediction" in result

    assert "species" in result