"""
data_loader.py

Loads the California Housing dataset and performs
the train-test split.
"""

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split


def load_data(test_size=0.2, random_state=42):
    """
    Load California Housing dataset.

    Returns:
        X_train, X_test, y_train, y_test,
        feature_names
    """

    housing = fetch_california_housing(as_frame=True)

    X = housing.data
    y = housing.target

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        X.columns.tolist()
    )