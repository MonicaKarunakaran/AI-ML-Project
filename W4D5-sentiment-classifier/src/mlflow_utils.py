"""
MLflow utility functions.
"""

from pathlib import Path

import mlflow


PROJECT_ROOT = Path(__file__).resolve().parent.parent

MLFLOW_DB = PROJECT_ROOT / "mlflow.db"

EXPERIMENT_NAME = "W4D5-Sentiment-Classifier"


def setup_mlflow():
    """
    Configure MLflow using a local SQLite backend.
    """

    tracking_uri = f"sqlite:///{MLFLOW_DB.as_posix()}"

    mlflow.set_tracking_uri(tracking_uri)

    mlflow.set_experiment(EXPERIMENT_NAME)

    return tracking_uri


def log_model_metrics(metrics: dict):
    """
    Log model metrics to the active MLflow run.
    """

    for name, value in metrics.items():

        if value is not None:
            mlflow.log_metric(
                name,
                float(value)
            )


def log_model_parameters(parameters: dict):
    """
    Log model parameters to the active MLflow run.
    """

    for name, value in parameters.items():
        mlflow.log_param(
            name,
            value
        )


def log_artifact(file_path):
    """
    Log a generated artifact.
    """

    file_path = Path(file_path)

    if file_path.exists():
        mlflow.log_artifact(
            str(file_path)
        )