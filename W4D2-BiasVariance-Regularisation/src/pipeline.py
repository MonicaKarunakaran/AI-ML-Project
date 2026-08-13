import os
import numpy as np
import pandas as pd
from sklearn.datasets import make_regression


def make_data(
    n_samples=500,
    n_features=10,
    noise=20,
    random_state=42,
):
    """
    Generate a synthetic regression dataset and save it as CSV.
    """

    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        noise=noise,
        random_state=random_state,
    )

    columns = [f"feature_{i+1}" for i in range(n_features)]

    df = pd.DataFrame(X, columns=columns)
    df["target"] = y

    os.makedirs("data", exist_ok=True)

    output_path = "data/synthetic.csv"
    df.to_csv(output_path, index=False)

    print("=" * 50)
    print("Synthetic dataset created successfully!")
    print(f"Location : {output_path}")
    print(f"Shape    : {df.shape}")
    print("=" * 50)

    return df


def main():
    make_data()


if __name__ == "__main__":
    main()