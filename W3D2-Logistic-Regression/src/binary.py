import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    ConfusionMatrixDisplay
)

from src.config import *


def train_binary_model(X_train, y_train):

    model = LogisticRegression(
        solver="lbfgs",
        max_iter=1000,
        random_state=RANDOM_STATE
    )

    model.fit(X_train, y_train)

    joblib.dump(model, BINARY_MODEL_PATH)

    return model


def evaluate_binary_model(model, X_test, y_test):

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    os.makedirs(RESULT_DIR, exist_ok=True)
    os.makedirs(FIGURE_DIR, exist_ok=True)

    metrics = pd.DataFrame({
        "Accuracy": [accuracy],
        "Precision": [precision],
        "Recall": [recall],
        "F1 Score": [f1],
        "ROC AUC": [roc_auc]
    })

    metrics.to_csv(
        os.path.join(RESULT_DIR, "binary_metrics.csv"),
        index=False
    )

    fpr, tpr, _ = roc_curve(y_test, y_prob)

    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, label=f"AUC={roc_auc:.2f}")
    plt.plot([0,1],[0,1],"--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()

    plt.savefig(
        os.path.join(FIGURE_DIR, "roc_binary.png")
    )

    plt.close()

    disp = ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred
    )

    disp.figure_.savefig(
        os.path.join(FIGURE_DIR, "confusion_binary.png")
    )

    plt.close()

    print(metrics)