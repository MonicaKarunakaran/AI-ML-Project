import mlflow
import mlflow.sklearn

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression


iris = load_iris()

X = iris.data
y = iris.target


model = LogisticRegression(max_iter=200)

model.fit(X,y)


mlflow.set_experiment(
    "Dummy Model Registry"
)


with mlflow.start_run():

    mlflow.sklearn.log_model(
        model,
        "iris_model"
    )


print("Dummy model registered")