"""
Data generation module
Creates an imbalanced binary classification dataset.
"""

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import pandas as pd


def load_data(
    samples=5000,
    features=20,
    random_state=42
):
    """
    Generate binary classification data.

    Returns:
        X_train, X_test, y_train, y_test
    """

    X, y = make_classification(
        n_samples=samples,
        n_features=features,
        n_informative=10,
        n_redundant=5,
        weights=[0.9, 0.1],
        random_state=random_state
    )

    X = pd.DataFrame(X)
    y = pd.Series(y)

    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=random_state,
        stratify=y
    )