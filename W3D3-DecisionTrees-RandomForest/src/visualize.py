"""
Visualization utilities
"""

import matplotlib.pyplot as plt

from sklearn.tree import plot_tree
from sklearn.metrics import ConfusionMatrixDisplay


from src.config import (
    TREE_PLOT_PATH,
    FEATURE_IMPORTANCE_PATH,
    CONFUSION_MATRIX_PATH
)



def visualize_tree(model, feature_names):

    plt.figure(figsize=(30,15))

    plot_tree(
        model,
        feature_names=feature_names,
        class_names=["Malignant","Benign"],
        filled=True
    )


    plt.savefig(
        TREE_PLOT_PATH,
        bbox_inches="tight"
    )

    plt.close()

    print("Decision Tree plot saved")



def feature_importance_plot(model, feature_names):

    importance = model.feature_importances_

    plt.figure(figsize=(10,6))

    plt.barh(
        feature_names,
        importance
    )

    plt.xlabel(
        "Importance"
    )

    plt.title(
        "Feature Importance"
    )

    plt.tight_layout()

    plt.savefig(
        FEATURE_IMPORTANCE_PATH
    )

    plt.close()

    print("Feature importance plot saved")



def confusion_matrix_plot(model,X_test,y_test):

    display = ConfusionMatrixDisplay.from_estimator(
        model,
        X_test,
        y_test
    )


    display.figure_.savefig(
        CONFUSION_MATRIX_PATH
    )


    plt.close()

    print("Confusion matrix saved")