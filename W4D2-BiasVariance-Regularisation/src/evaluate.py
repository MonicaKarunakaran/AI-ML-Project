import os
import pandas as pd
import matplotlib.pyplot as plt

from joblib import load

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)


DATA_PATH = "data/synthetic.csv"


def load_dataset():
    """Load and split the dataset."""

    df = pd.read_csv(DATA_PATH)

    X = df.drop("target", axis=1)
    y = df["target"]

    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )


def evaluate_model(model_path, model_name):
    """
    Evaluate a saved model.
    """

    X_train, X_test, y_train, y_test = load_dataset()

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = load(model_path)

    predictions = model.predict(X_test)

    mse = mean_squared_error(y_test, predictions)
    rmse = mse ** 0.5
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print("\n" + "=" * 50)
    print(model_name)
    print("=" * 50)
    print(f"MSE  : {mse:.3f}")
    print(f"RMSE : {rmse:.3f}")
    print(f"MAE  : {mae:.3f}")
    print(f"R²   : {r2:.3f}")

    return y_test, predictions


def plot_predictions(actual, predicted, title, filename):

    os.makedirs("plots", exist_ok=True)

    plt.figure(figsize=(8, 6))

    plt.scatter(actual, predicted)

    plt.plot(
        [actual.min(), actual.max()],
        [actual.min(), actual.max()],
    )

    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title(title)

    plt.tight_layout()

    plt.savefig(f"plots/{filename}")

    plt.close()


def compare_models():

    actual1, pred1 = evaluate_model(
        "models/linear_model.pkl",
        "Linear Regression",
    )

    plot_predictions(
        actual1,
        pred1,
        "Linear Regression Predictions",
        "linear_predictions.png",
    )

    actual2, pred2 = evaluate_model(
        "models/ridge_model.pkl",
        "Ridge Regression",
    )

    plot_predictions(
        actual2,
        pred2,
        "Ridge Regression Predictions",
        "ridge_predictions.png",
    )

    linear_error = abs(actual1 - pred1).mean()
    ridge_error = abs(actual2 - pred2).mean()

    plt.figure(figsize=(6, 5))

    plt.bar(
        ["Linear", "Ridge"],
        [linear_error, ridge_error],
    )

    plt.ylabel("Mean Absolute Error")
    plt.title("Bias-Variance Comparison")

    plt.tight_layout()

    plt.savefig("plots/bias_variance_comparison.png")

    plt.close()

    print("\nPlots saved in the 'plots/' folder.")


def main():
    compare_models()


if __name__ == "__main__":
    main()