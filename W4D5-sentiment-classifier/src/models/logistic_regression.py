"""
Logistic Regression training and evaluation.
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import mlflow
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)


def train_logistic_regression(
    X_train,
    y_train,
    C: float = 1.0,
):
    """
    Train Logistic Regression.
    """

    model = LogisticRegression(
        C=C,
        max_iter=1000,
        random_state=42,
    )

    model.fit(X_train, y_train)

    return model


def evaluate_logistic_regression(
    model,
    X_test,
    y_test,
    save_plots: bool = True,
):
    """
    Evaluate Logistic Regression.
    """

    predictions = model.predict(X_test)

    probabilities = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test,
        predictions,
        zero_division=0,
    )
    recall = recall_score(
        y_test,
        predictions,
        zero_division=0,
    )
    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0,
    )
    roc_auc = roc_auc_score(
        y_test,
        probabilities,
    )

    print("\n" + "=" * 60)
    print("LOGISTIC REGRESSION")
    print("=" * 60)

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            target_names=["negative", "positive"],
            zero_division=0,
        )
    )

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc,
    }

    if save_plots:
        plot_confusion_matrix(
            y_test,
            predictions,
            "logistic_confusion_matrix.png",
        )

        plot_roc_curve(
            y_test,
            probabilities,
            "logistic_roc_curve.png",
        )

    return metrics


def plot_confusion_matrix(
    y_true,
    predictions,
    filename: str,
):
    """
    Save confusion matrix plot.
    """

    cm = confusion_matrix(y_true, predictions)

    plt.figure(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=["Negative", "Positive"],
        yticklabels=["Negative", "Positive"],
    )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Logistic Regression - Confusion Matrix")

    plt.tight_layout()

    output_path = OUTPUT_DIR / filename

    plt.savefig(output_path)

    plt.close()

    print(f"Saved: {output_path}")


def plot_roc_curve(
    y_true,
    probabilities,
    filename: str,
):
    """
    Save ROC-AUC curve.
    """

    fpr, tpr, _ = roc_curve(
        y_true,
        probabilities,
    )

    auc = roc_auc_score(
        y_true,
        probabilities,
    )

    plt.figure(figsize=(7, 5))

    plt.plot(
        fpr,
        tpr,
        label=f"Logistic Regression (AUC = {auc:.3f})",
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Logistic Regression - ROC Curve")
    plt.legend()

    plt.tight_layout()

    output_path = OUTPUT_DIR / filename

    plt.savefig(output_path)

    plt.close()

    print(f"Saved: {output_path}")


def save_model(
    model,
    vectorizer,
):
    """
    Save Logistic Regression model and TF-IDF vectorizer.
    """

    model_path = OUTPUT_DIR / "logistic_regression_model.joblib"
    vectorizer_path = OUTPUT_DIR / "tfidf_vectorizer.joblib"

    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)

    return model_path, vectorizer_path