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
    ConfusionMatrixDisplay
)

from src.config import *


def train_multiclass_model(X_train, y_train):

    model = LogisticRegression(
        solver="lbfgs",
        max_iter=1000,
        random_state=RANDOM_STATE
    )

    model.fit(X_train, y_train)

    joblib.dump(model, MULTI_MODEL_PATH)

    return model


def evaluate_multiclass_model(model, X_test, y_test):

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        average="macro"
    )

    recall = recall_score(
        y_test,
        y_pred,
        average="macro"
    )

    f1 = f1_score(
        y_test,
        y_pred,
        average="macro"
    )

    os.makedirs(RESULT_DIR, exist_ok=True)
    os.makedirs(FIGURE_DIR, exist_ok=True)

    metrics = pd.DataFrame({
        "Accuracy":[accuracy],
        "Precision":[precision],
        "Recall":[recall],
        "F1 Score":[f1]
    })

    metrics.to_csv(
        os.path.join(RESULT_DIR,"multi_metrics.csv"),
        index=False
    )

    disp = ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred
    )

    disp.figure_.savefig(
        os.path.join(FIGURE_DIR,"confusion_multi.png")
    )

    plt.close()

    print(metrics)