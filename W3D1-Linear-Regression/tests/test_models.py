from src.data_loader import load_data
from src.preprocessing import scale_data
from src.models import (
    train_linear,
    train_ridge,
    train_lasso
)


def test_linear_regression_training():

    X_train, X_test, y_train, y_test, feature_names = load_data()

    X_train_scaled, X_test_scaled = scale_data(
        X_train,
        X_test
    )

    model = train_linear(
        X_train_scaled,
        y_train
    )

    # Check model has learned coefficients
    assert model.coef_ is not None

    # Check number of coefficients matches features
    assert len(model.coef_) == len(feature_names)



def test_ridge_training():

    X_train, X_test, y_train, y_test, feature_names = load_data()

    X_train_scaled, X_test_scaled = scale_data(
        X_train,
        X_test
    )

    model = train_ridge(
        X_train_scaled,
        y_train
    )

    assert model.coef_ is not None



def test_lasso_training():

    X_train, X_test, y_train, y_test, feature_names = load_data()

    X_train_scaled, X_test_scaled = scale_data(
        X_train,
        X_test
    )

    model = train_lasso(
        X_train_scaled,
        y_train
    )

    assert model.coef_ is not None