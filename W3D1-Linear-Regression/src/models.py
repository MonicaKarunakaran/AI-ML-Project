"""
models.py

Train Linear Regression,
Ridge Regression,
and Lasso Regression.
"""

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso


def train_linear(X_train, y_train):

    model = LinearRegression()

    model.fit(X_train, y_train)

    return model


def train_ridge(X_train, y_train):

    model = Ridge(alpha=1.0)

    model.fit(X_train, y_train)

    return model


def train_lasso(X_train, y_train):

    model = Lasso(alpha=0.1)

    model.fit(X_train, y_train)

    return model