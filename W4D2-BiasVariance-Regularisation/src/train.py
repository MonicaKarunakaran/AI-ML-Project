import os
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

from joblib import dump

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "synthetic.csv"


def load_data():
    df = pd.read_csv(DATA_PATH)

    X = df.drop("target", axis=1)
    y = df["target"]

    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )


def evaluate(model, X_train, X_test, y_train, y_test):
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    metrics = {
        "train_mse": mean_squared_error(y_train, train_pred),
        "test_mse": mean_squared_error(y_test, test_pred),
        "train_mae": mean_absolute_error(y_train, train_pred),
        "test_mae": mean_absolute_error(y_test, test_pred),
        "train_r2": r2_score(y_train, train_pred),
        "test_r2": r2_score(y_test, test_pred),
    }

    return metrics


def train_linear():

    X_train, X_test, y_train, y_test = load_data()

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = LinearRegression()

    model.fit(X_train, y_train)

    metrics = evaluate(
        model,
        X_train,
        X_test,
        y_train,
        y_test,
    )

    mlflow.start_run(run_name="LinearRegression")

    mlflow.log_param("model", "LinearRegression")

    for key, value in metrics.items():
        mlflow.log_metric(key, value)

    os.makedirs("models", exist_ok=True)

    dump(model, "models/linear_model.pkl")

    mlflow.sklearn.log_model(
        model,
        artifact_path="linear_model",
    )

    mlflow.end_run()

    print("\nLinear Regression Results")

    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    return metrics


def train_ridge(alpha=1.0):

    X_train, X_test, y_train, y_test = load_data()

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = Ridge(alpha=alpha)

    model.fit(X_train, y_train)

    metrics = evaluate(
        model,
        X_train,
        X_test,
        y_train,
        y_test,
    )

    mlflow.start_run(run_name="RidgeRegression")

    mlflow.log_param("model", "Ridge")

    mlflow.log_param("alpha", alpha)

    for key, value in metrics.items():
        mlflow.log_metric(key, value)

    os.makedirs("models", exist_ok=True)

    dump(model, "models/ridge_model.pkl")

    mlflow.sklearn.log_model(
        model,
        artifact_path="ridge_model",
    )

    mlflow.end_run()

    print("\nRidge Regression Results")

    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    return metrics


def main():

    mlflow.set_experiment("Bias-Variance-Regularisation")

    print("=" * 60)
    print("Training Linear Regression")
    print("=" * 60)

    train_linear()

    print("\n")

    print("=" * 60)
    print("Training Ridge Regression")
    print("=" * 60)

    train_ridge(alpha=10)


if __name__ == "__main__":
    main()