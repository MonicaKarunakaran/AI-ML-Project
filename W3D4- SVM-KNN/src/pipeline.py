import os

import joblib
import mlflow
import mlflow.sklearn

from models import SVMModel, KNNModel
from utils import (
    load_data,
    scale_data,
    log_metrics,
    save_confusion_matrix,
)
from evaluation import evaluate_model


os.makedirs("models", exist_ok=True)
os.makedirs("plots", exist_ok=True)

mlflow.set_experiment("W3D4_SVM_KNN")

X_train, X_test, y_train, y_test = load_data()

X_train, X_test = scale_data(X_train, X_test)


# -------------------------
# SVM
# -------------------------

with mlflow.start_run(run_name="SVM"):

    svm = SVMModel()

    svm.train(X_train, y_train)

    predictions = svm.predict(X_test)

    accuracy, precision, recall, f1 = evaluate_model(
        y_test,
        predictions,
    )

    log_metrics(
        "SVM",
        accuracy,
        precision,
        recall,
        f1,
    )

    mlflow.sklearn.log_model(
        svm.model,
        "svm_model",
    )

    joblib.dump(
        svm.model,
        "models/svm_model.pkl",
    )

    save_confusion_matrix(
        y_test,
        predictions,
        "plots/svm_confusion_matrix.png",
    )

    print(f"SVM Accuracy : {accuracy:.4f}")


# -------------------------
# KNN
# -------------------------

with mlflow.start_run(run_name="KNN"):

    knn = KNNModel()

    knn.train(X_train, y_train)

    predictions = knn.predict(X_test)

    accuracy, precision, recall, f1 = evaluate_model(
        y_test,
        predictions,
    )

    log_metrics(
        "KNN",
        accuracy,
        precision,
        recall,
        f1,
    )

    mlflow.sklearn.log_model(
        knn.model,
        "knn_model",
    )

    joblib.dump(
        knn.model,
        "models/knn_model.pkl",
    )

    save_confusion_matrix(
        y_test,
        predictions,
        "plots/knn_confusion_matrix.png",
    )

    print(f"KNN Accuracy : {accuracy:.4f}")

print("\nTraining Completed Successfully.")