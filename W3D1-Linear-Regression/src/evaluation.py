"""
evaluation.py

Evaluation metrics
and plotting utilities.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)


def evaluate_model(model, X_test, y_test):

    predictions = model.predict(X_test)

    mse = mean_squared_error(y_test, predictions)

    rmse = np.sqrt(mse)

    mae = mean_absolute_error(y_test, predictions)

    r2 = r2_score(y_test, predictions)

    return {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
        "Predictions": predictions
    }


def plot_predictions(y_test, predictions, model_name):

    os.makedirs("plots", exist_ok=True)

    plt.figure(figsize=(6,6))

    plt.scatter(y_test, predictions)

    plt.xlabel("Actual")

    plt.ylabel("Predicted")

    plt.title(f"{model_name} - Predicted vs Actual")

    plt.savefig(
        f"plots/{model_name.lower()}_pred_vs_actual.png"
    )

    plt.close()


def plot_residuals(y_test, predictions, model_name):

    residuals = y_test - predictions

    plt.figure(figsize=(6,4))

    plt.scatter(predictions, residuals)

    plt.axhline(0, linestyle="--")

    plt.xlabel("Predicted")

    plt.ylabel("Residual")

    plt.title(f"{model_name} Residual Plot")

    plt.savefig(
        f"plots/{model_name.lower()}_residuals.png"
    )

    plt.close()


def save_results(results):

    os.makedirs("results", exist_ok=True)

    df = pd.DataFrame(results)

    df.to_csv(
        "results/model_comparison.csv",
        index=False
    )

    print(df)