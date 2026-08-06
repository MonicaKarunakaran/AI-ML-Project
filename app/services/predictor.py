import pandas as pd

from app.services.model_loader import get_model, get_scaler


SPECIES = {
    0: "Setosa",
    1: "Versicolor",
    2: "Virginica"
}


def predict(features: pd.DataFrame):

    model = get_model()
    scaler = get_scaler()

    scaled_features = scaler.transform(features)

    prediction = model.predict(scaled_features)[0]

    species = SPECIES[int(prediction)]

    return {
        "prediction": int(prediction),
        "species": species
    }