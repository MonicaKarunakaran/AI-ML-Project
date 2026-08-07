"""
Random Forest training and evaluation.
"""

from pathlib import Path

import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)


def train_random_forest(
    X_train,
    y_train,
    n_estimators: int = 200,
    max_depth=None,
):
    """
    Train Random Forest classifier.
    """

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    return model


def evaluate_random_forest(
    model,
    X_test,
    y_test,
):
    """
    Evaluate Random Forest.
    """

    predictions = model.predict(X_test)

    probabilities = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

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
    print("RANDOM FOREST")
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

    return metrics


def save_model(
    model,
):
    """
    Save Random Forest model.
    """

    model_path = OUTPUT_DIR / "random_forest_model.joblib"

    joblib.dump(
        model,
        model_path,
    )

    return model_path