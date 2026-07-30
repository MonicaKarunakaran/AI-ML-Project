"""
Data loading and preprocessing.
"""

import os
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from src.config import DATASET_PATH, TEST_SIZE, RANDOM_STATE


def load_data():
    """
    Load the Breast Cancer dataset.
    Save it as a CSV if it does not already exist.
    """

    if not os.path.exists(DATASET_PATH):
        dataset = load_breast_cancer(as_frame=True)
        df = dataset.frame

        os.makedirs("data", exist_ok=True)
        df.to_csv(DATASET_PATH, index=False)
        print(f"Dataset saved to {DATASET_PATH}")

    df = pd.read_csv(DATASET_PATH)

    X = df.drop("target", axis=1)
    y = df["target"]

    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data()

    print("Dataset Loaded Successfully!")
    print(f"Training Samples : {len(X_train)}")
    print(f"Testing Samples  : {len(X_test)}")