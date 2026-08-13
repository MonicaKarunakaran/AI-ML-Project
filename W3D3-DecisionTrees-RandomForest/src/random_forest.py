"""
Random Forest Model Training with MLflow
"""

import joblib

from sklearn.ensemble import RandomForestClassifier

from src.data_preprocessing import load_data
from src.evaluate import evaluate_model
from src.mlflow_logger import setup_mlflow, log_model_metrics
from src.config import (
    RANDOM_FOREST_MODEL_PATH,
    RANDOM_STATE
)


def train_random_forest():

    # Load data
    X_train, X_test, y_train, y_test = load_data()


    # Create Random Forest model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=RANDOM_STATE
    )


    # Train model
    model.fit(
        X_train,
        y_train
    )


    # Evaluate
    metrics, predictions = evaluate_model(
        model,
        X_test,
        y_test
    )


    # Save model
    joblib.dump(
        model,
        RANDOM_FOREST_MODEL_PATH
    )


    print("\nRandom Forest model saved!")


    # MLflow logging

    params = {
        "n_estimators":100,
        "max_depth":5
    }


    log_model_metrics(
        model,
        "Random_Forest",
        metrics,
        params
    )


    return model



if __name__ == "__main__":

    setup_mlflow()

    train_random_forest()