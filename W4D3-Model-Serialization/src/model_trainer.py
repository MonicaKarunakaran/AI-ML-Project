import os
import pickle
import joblib
import mlflow
import mlflow.sklearn

from sklearn.linear_model import LogisticRegression

from utils import load_data, preprocess_data
from evaluate import evaluate_model

ARTIFACT_PATH = "artifacts"

os.makedirs(
    ARTIFACT_PATH,
    exist_ok=True
)

def train_model():

    print("Loading dataset...")
    df = load_data()
    print("Preprocessing data...")
    (
        X_train,
        X_test,
        y_train,
        y_test,
        scaler
    ) = preprocess_data(df)
    print("Training model...")
    model = LogisticRegression(
        max_iter=200
    )

    model.fit(
        X_train,
        y_train
    )

    print("Evaluating model...")

    metrics = evaluate_model(
        model,
        X_test,
        y_test
    )

    print("\nModel Metrics")

    for key,value in metrics.items():

        print(
            f"{key}: {value:.4f}"
        )

    mlflow.set_experiment(
        "W4D3_Model_Serialization"
    )

    with mlflow.start_run():

        mlflow.log_param(
            "model",
            "Logistic Regression"
        )

        mlflow.log_param(
            "max_iter",
            200
        )

        for key,value in metrics.items():

            mlflow.log_metric(
                key,
                value
            )
        mlflow.sklearn.log_model(
            model,
            "model"
        )

    joblib_path = os.path.join(
        ARTIFACT_PATH,
        "model_joblib.pkl"
    )
    joblib.dump(
        model,
        joblib_path
    )
    print(
        "Model saved using joblib"
    )

    pickle_path = os.path.join(
        ARTIFACT_PATH,
        "model_pickle.pkl"
    )

    with open(
        pickle_path,
        "wb"
    ) as file:
        pickle.dump(
            model,
            file
        )
    print(
        "Model saved using pickle"
    )

    scaler_path = os.path.join(
        ARTIFACT_PATH,
        "scaler.pkl"
    )
    joblib.dump(
        scaler,
        scaler_path
    )
    print(
        "Scaler saved"
    )
    print("\nTraining completed successfully")

if __name__ == "__main__":
    train_model()