"""
linear_regression.py

Main script for W3D1 - Linear Regression

Workflow:
1. Load dataset
2. Preprocess data
3. Train Linear, Ridge, and Lasso models
4. Evaluate models
5. Save plots
6. Save comparison results
"""

from src.data_loader import load_data
from src.preprocessing import scale_data
from src.models import (
    train_linear,
    train_ridge,
    train_lasso
)
from src.evaluation import (
    evaluate_model,
    plot_predictions,
    plot_residuals,
    save_results
)


def print_coefficients(model, feature_names, model_name):
    """Print model coefficients and intercept."""

    print("\n" + "=" * 60)
    print(f"{model_name}")
    print("=" * 60)

    print(f"\nIntercept: {model.intercept_:.4f}\n")

    print("Coefficients:")

    for feature, coef in zip(feature_names, model.coef_):
        print(f"{feature:<20} : {coef:.4f}")


def main():

    print("=" * 60)
    print("W3D1 - LINEAR REGRESSION")
    print("=" * 60)

    X_train, X_test, y_train, y_test, feature_names = load_data()

    print("\nDataset Loaded Successfully")
    print(f"Training Samples : {len(X_train)}")
    print(f"Testing Samples  : {len(X_test)}")

    X_train_scaled, X_test_scaled = scale_data(
        X_train,
        X_test
    )

    models = {
        "Linear": train_linear(
            X_train_scaled,
            y_train
        ),
        "Ridge": train_ridge(
            X_train_scaled,
            y_train
        ),
        "Lasso": train_lasso(
            X_train_scaled,
            y_train
        )
    }

    results = []

    for model_name, model in models.items():

        print_coefficients(
            model,
            feature_names,
            model_name
        )

        metrics = evaluate_model(
            model,
            X_test_scaled,
            y_test
        )

        print("\nEvaluation Metrics")
        print("-" * 30)

        print(f"MSE  : {metrics['MSE']:.4f}")
        print(f"RMSE : {metrics['RMSE']:.4f}")
        print(f"MAE  : {metrics['MAE']:.4f}")
        print(f"R²   : {metrics['R2']:.4f}")

        plot_predictions(
            y_test,
            metrics["Predictions"],
            model_name
        )

        plot_residuals(
            y_test,
            metrics["Predictions"],
            model_name
        )

        if model_name == "Linear":
            shrinkage = "None"
        elif model_name == "Ridge":
            shrinkage = "Moderate"
        else:
            shrinkage = "High"

        results.append({
            "Model": model_name,
            "MSE": round(metrics["MSE"], 4),
            "RMSE": round(metrics["RMSE"], 4),
            "MAE": round(metrics["MAE"], 4),
            "R²": round(metrics["R2"], 4),
            "Coefficient Shrinkage": shrinkage
        })

    save_results(results)

    print("\n" + "=" * 60)
    print("Project Completed Successfully!")
    print("=" * 60)

    print("\nGenerated Files:")
    print("✔ plots/")
    print("   - linear_pred_vs_actual.png")
    print("   - linear_residuals.png")
    print("   - ridge_pred_vs_actual.png")
    print("   - ridge_residuals.png")
    print("   - lasso_pred_vs_actual.png")
    print("   - lasso_residuals.png")

    print("\n✔ results/model_comparison.csv")


if __name__ == "__main__":
    main()