import os
import pandas as pd
from src.loader import load_binary_dataset, load_multiclass_dataset
from src.preprocess import preprocess_data
from src.binary import train_binary_model, evaluate_binary_model
from src.multiclass import train_multiclass_model, evaluate_multiclass_model
from src import mlflow_logger
from src.config import *

def main():
    print("=== Starting Logistic Regression Pipeline ===")

    # ----------------------------------------------------
    # 1. Binary Classification Pipeline (Breast Cancer)
    # ----------------------------------------------------
    print("\n[1/2] Running Binary Classification Pipeline...")
    X_train_bin, X_test_bin, y_train_bin, y_test_bin = load_binary_dataset()

    # Preprocess
    X_train_bin_scaled, X_test_bin_scaled = preprocess_data(X_train_bin, X_test_bin)

    # Train
    binary_model = train_binary_model(X_train_bin_scaled, y_train_bin)

    # Evaluate
    evaluate_binary_model(binary_model, X_test_bin_scaled, y_test_bin)

    # MLflow Logging for Binary Model
    with mlflow_logger.start_run():
        mlflow_logger.log_parameters()

        # Read binary metrics CSV to log to MLflow
        binary_metrics_df = pd.read_csv(os.path.join(RESULT_DIR, "binary_metrics.csv"))
        binary_metrics = binary_metrics_df.iloc[0].to_dict()
        mlflow_logger.log_metrics(binary_metrics)

        # Log artifacts (plots and models)
        mlflow_logger.log_artifact(os.path.join(FIGURE_DIR, "roc_binary.png"))
        mlflow_logger.log_artifact(os.path.join(FIGURE_DIR, "confusion_binary.png"))
        mlflow_logger.log_artifact(BINARY_MODEL_PATH)
        mlflow_logger.log_artifact(SCALER_PATH)

    print("Binary Classification completed and logged to MLflow.")

    # ----------------------------------------------------
    # 2. Multiclass Classification Pipeline (Iris)
    # ----------------------------------------------------
    print("\n[2/2] Running Multiclass Classification Pipeline...")
    X_train_multi, X_test_multi, y_train_multi, y_test_multi = load_multiclass_dataset()

    # Preprocess using new scaler instance
    X_train_multi_scaled, X_test_multi_scaled = preprocess_data(X_train_multi, X_test_multi)

    # Train
    multi_model = train_multiclass_model(X_train_multi_scaled, y_train_multi)

    # Evaluate
    evaluate_multiclass_model(multi_model, X_test_multi_scaled, y_test_multi)

    # MLflow Logging for Multiclass Model
    with mlflow_logger.start_run():
        mlflow_logger.log_parameters()

        # Read multiclass metrics CSV to log to MLflow
        multi_metrics_df = pd.read_csv(os.path.join(RESULT_DIR, "multi_metrics.csv"))
        multi_metrics = multi_metrics_df.iloc[0].to_dict()
        mlflow_logger.log_metrics(multi_metrics)

        # Log artifacts
        mlflow_logger.log_artifact(os.path.join(FIGURE_DIR, "confusion_multi.png"))
        mlflow_logger.log_artifact(MULTI_MODEL_PATH)

    print("Multiclass Classification completed and logged to MLflow.")
    print("\n=== Pipeline Execution Finished Successfully ===")

if __name__ == "__main__":
    main()