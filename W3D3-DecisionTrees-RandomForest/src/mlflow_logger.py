"""
MLflow logging utility functions.
"""

import mlflow
import mlflow.sklearn

from src.config import MLFLOW_EXPERIMENT


def setup_mlflow():
    """
    Configure MLflow experiment.
    """

    mlflow.set_experiment(MLFLOW_EXPERIMENT)


def log_model_metrics(model, model_name, metrics, params):
    """
    Log model parameters, metrics, and model artifact.

    Args:
        model: trained sklearn model
        model_name: name of model
        metrics: dictionary of evaluation metrics
        params: dictionary of model parameters
    """

    with mlflow.start_run(run_name=model_name):

        # Log parameters
        mlflow.log_params(params)

        # Log metrics
        mlflow.log_metrics(metrics)

        # Log sklearn model
        mlflow.sklearn.log_model(
            model,
            artifact_path=model_name
        )

        print(f"{model_name} logged successfully in MLflow")