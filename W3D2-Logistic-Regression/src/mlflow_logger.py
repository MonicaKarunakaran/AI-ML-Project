import mlflow


def start_run():

    mlflow.set_experiment("W3D2 Logistic Regression")

    return mlflow.start_run()


def log_parameters():

    mlflow.log_param("solver", "lbfgs")
    mlflow.log_param("max_iter", 1000)
    mlflow.log_param("threshold", 0.5)


def log_metrics(metrics):

    for key, value in metrics.items():
        mlflow.log_metric(key, value)


def log_artifact(path):

    mlflow.log_artifact(path)