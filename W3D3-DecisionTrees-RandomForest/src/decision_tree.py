"""
Decision Tree Model Training with MLflow
"""

import joblib
import mlflow

from sklearn.tree import DecisionTreeClassifier

from src.data_preprocessing import load_data
from src.evaluate import evaluate_model
from src.mlflow_logger import setup_mlflow, log_model_metrics
from src.config import (
    DECISION_TREE_MODEL_PATH,
    RANDOM_STATE
)


def train_decision_tree():

    # Load data
    X_train, X_test, y_train, y_test = load_data()

    # Create model
    model = DecisionTreeClassifier(
        criterion="gini",
        max_depth=5,
        random_state=RANDOM_STATE
    )

    # Train
    model.fit(X_train, y_train)

    # Evaluate
    metrics, predictions = evaluate_model(
        model,
        X_test,
        y_test
    )

    # Save model
    joblib.dump(
        model,
        DECISION_TREE_MODEL_PATH
    )

    print("\nDecision Tree model saved!")

    # MLflow logging
    params = {
        "criterion": "gini",
        "max_depth": 5
    }

    log_model_metrics(
        model,
        "Decision_Tree",
        metrics,
        params
    )

    return model


if __name__ == "__main__":

    setup_mlflow()

    train_decision_tree()