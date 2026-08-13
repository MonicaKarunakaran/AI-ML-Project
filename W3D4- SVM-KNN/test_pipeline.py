from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.models import SVMModel


def test_svm():

    data = load_breast_cancer()

    X_train, X_test, y_train, y_test = train_test_split(
        data.data,
        data.target,
        test_size=0.2,
        random_state=42,
    )

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = SVMModel()

    model.train(X_train, y_train)

    predictions = model.predict(X_test)

    assert len(predictions) == len(y_test)