"""
preprocessing.py

Contains preprocessing utilities.
"""

from sklearn.preprocessing import StandardScaler


def scale_data(X_train, X_test):
    """
    Scale features using StandardScaler.

    Returns:
        X_train_scaled,
        X_test_scaled
    """

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)

    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled