import joblib
import pandas as pd
import os

MODEL_PATH = "artifacts/model_joblib.pkl"
SCALER_PATH = "artifacts/scaler.pkl"

def load_model():

    model = joblib.load(
        MODEL_PATH
    )

    scaler = joblib.load(
        SCALER_PATH
    )
    return model, scaler

def predict():

    print("Loading saved model...")
    model, scaler = load_model()
    print("Model loaded successfully")
    new_data = pd.DataFrame(
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
    new_data_scaled = scaler.transform(
        new_data
    )
    prediction = model.predict(
        new_data_scaled
    )
    print(
        "Prediction:",
        prediction[0]
    )
    return prediction[0]

if __name__ == "__main__":

    predict()