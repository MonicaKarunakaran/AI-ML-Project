"""
Model evaluation functions
"""

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve
)

import matplotlib.pyplot as plt


def calculate_metrics(y_true, y_pred, y_prob):

    metrics = {

        "precision":
        precision_score(y_true, y_pred),

        "recall":
        recall_score(y_true, y_pred),

        "f1_score":
        f1_score(y_true, y_pred),

        "auc_roc":
        roc_auc_score(y_true, y_prob)

    }

    return metrics



def plot_roc_curve(y_true, y_prob):

    fpr, tpr, _ = roc_curve(
        y_true,
        y_prob
    )

    plt.figure(figsize=(7,5))

    plt.plot(
        fpr,
        tpr
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

    plt.title("ROC Curve")

    plt.savefig(
        "plots/roc_curve.png"
    )

    plt.close()