import os
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from src.config import *


def load_binary_dataset():

    dataset = load_breast_cancer()

    df = pd.DataFrame(
        dataset.data,
        columns=dataset.feature_names
    )

    df["target"] = dataset.target

    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

    train = X_train.copy()
    train["target"] = y_train.values

    test = X_test.copy()
    test["target"] = y_test.values

    train.to_csv(
        os.path.join(PROCESSED_DATA_DIR, "binary_train.csv"),
        index=False,
    )

    test.to_csv(
        os.path.join(PROCESSED_DATA_DIR, "binary_test.csv"),
        index=False,
    )

    return X_train, X_test, y_train, y_test


def load_multiclass_dataset():

    iris = load_iris()

    df = pd.DataFrame(
        iris.data,
        columns=iris.feature_names
    )

    df["target"] = iris.target

    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    return X_train, X_test, y_train, y_test