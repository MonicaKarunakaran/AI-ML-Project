import os
import joblib
import mlflow
import mlflow.sklearn

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

os.makedirs("models", exist_ok=True)

iris = load_iris()

X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

mlflow.set_experiment("W4D4-FastAPI-Model-Serving")


with mlflow.start_run():

    model = LogisticRegression(max_iter=200)

    model.fit(X_train_scaled, y_train)

    predictions = model.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test,
        predictions,
        average="weighted",
    )
    recall = recall_score(
        y_test,
        predictions,
        average="weighted",
    )
    f1 = f1_score(
        y_test,
        predictions,
        average="weighted",
    )

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

    mlflow.log_param("model", "Logistic Regression")
    mlflow.log_param("dataset", "Iris")

    mlflow.sklearn.log_model(
        sk_model=model,
        name="iris_model",
    )

joblib.dump(model, "models/iris_model.pkl")

joblib.dump(scaler, "models/scaler.pkl")


print("=" * 50)
print("Model Training Completed")
print("=" * 50)
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print()
print("Model saved in models/iris_model.pkl")
print("Scaler saved in models/scaler.pkl")