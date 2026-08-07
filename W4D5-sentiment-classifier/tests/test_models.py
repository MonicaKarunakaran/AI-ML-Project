"""
Tests for Logistic Regression and Random Forest.
"""

from sklearn.feature_extraction.text import TfidfVectorizer

from src.models.logistic_regression import (
    train_logistic_regression,
)

from src.models.random_forest import (
    train_random_forest,
)


def create_test_data():

    texts = [
        "I love this product",
        "Amazing experience",
        "Very good service",
        "Excellent quality",
        "I hate this product",
        "Terrible experience",
        "Very bad service",
        "Worst quality",
    ]

    labels = [
        1,
        1,
        1,
        1,
        0,
        0,
        0,
        0,
    ]

    vectorizer = TfidfVectorizer()

    features = vectorizer.fit_transform(
        texts
    )

    return features, labels


def test_logistic_regression():

    X, y = create_test_data()

    model = train_logistic_regression(
        X,
        y,
    )

    predictions = model.predict(X)

    assert len(predictions) == len(y)


def test_random_forest():

    X, y = create_test_data()

    model = train_random_forest(
        X,
        y,
        n_estimators=10,
    )

    predictions = model.predict(X)

    assert len(predictions) == len(y)