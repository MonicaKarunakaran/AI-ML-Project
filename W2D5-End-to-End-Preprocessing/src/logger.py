import mlflow


def log_pipeline():

    with mlflow.start_run():

        mlflow.log_param(
            "imputer",
            "SimpleImputer"
        )

        mlflow.log_param(
            "scaler",
            "StandardScaler"
        )

        mlflow.log_param(
            "encoder",
            "OneHotEncoder"
        )


        mlflow.log_artifact(
            "output/titanic_preprocessed.csv"
        )


    print("MLflow logging completed")