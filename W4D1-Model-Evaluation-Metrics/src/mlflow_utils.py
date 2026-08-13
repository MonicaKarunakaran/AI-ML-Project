"""
MLflow utility functions
"""

import mlflow
import mlflow.sklearn



def setup_mlflow():

    mlflow.set_experiment(
        "W4D1-Precision-Recall-AUC"
    )



def log_results(
        model,
        metrics
):

    for name,value in metrics.items():

        mlflow.log_metric(
            name,
            value
        )


    mlflow.sklearn.log_model(
        model,
        "model"
    )