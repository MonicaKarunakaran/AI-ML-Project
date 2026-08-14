import os

import matplotlib.pyplot as plt
import shap


def run_shap(
    model,
    X_train,
    X_test,
    out_dir="shap_plots"
):
    """
    Generate SHAP explanations for a Logistic Regression model.

    Creates:
        - shap_summary.png
        - shap_waterfall.png
    """

    os.makedirs(out_dir, exist_ok=True)

    print("Generating SHAP explanations...")

    # SHAP automatically detects the linear model.
    explainer = shap.Explainer(model, X_train)

    shap_values = explainer(X_test)

    # Summary plot
    plt.figure()

    shap.summary_plot(
        shap_values,
        X_test,
        show=False
    )

    plt.tight_layout()

    summary_path = os.path.join(
        out_dir,
        "shap_summary.png"
    )

    plt.savefig(
        summary_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # Waterfall plot for first test instance
    plt.figure()

    shap.plots.waterfall(
        shap_values[0],
        show=False
    )

    plt.tight_layout()

    waterfall_path = os.path.join(
        out_dir,
        "shap_waterfall.png"
    )

    plt.savefig(
        waterfall_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved: {summary_path}")
    print(f"Saved: {waterfall_path}")

    return shap_values