"""
Data loading and cleaning utilities for the sentiment classifier.
"""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "sentiment.csv"


def load_sentiment_data(
    file_path: str | Path = DEFAULT_DATA_PATH,
) -> pd.DataFrame:
    """
    Load the sentiment dataset from a CSV file.

    Expected columns:
        text
        sentiment

    Returns:
        Cleaned pandas DataFrame.
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {file_path}. "
            "Place sentiment.csv inside the data/ directory."
        )

    df = pd.read_csv(file_path)

    required_columns = {"text", "sentiment"}

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}. "
            "Dataset must contain 'text' and 'sentiment'."
        )

    df = df[["text", "sentiment"]].copy()

    # Remove missing values
    df = df.dropna(subset=["text", "sentiment"])

    # Convert text to string
    df["text"] = df["text"].astype(str).str.strip()

    # Normalize labels
    df["sentiment"] = (
        df["sentiment"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    # Remove empty text
    df = df[df["text"] != ""]

    # Keep only binary sentiment labels
    valid_labels = {"positive", "negative"}

    df = df[df["sentiment"].isin(valid_labels)]

    # Convert sentiment to binary target
    df["label"] = df["sentiment"].map(
        {
            "negative": 0,
            "positive": 1,
        }
    )

    df = df.reset_index(drop=True)

    if df.empty:
        raise ValueError("Dataset is empty after cleaning.")

    if df["label"].nunique() < 2:
        raise ValueError(
            "Dataset must contain both positive and negative samples."
        )

    return df


def get_features_and_target(
    df: pd.DataFrame,
):
    """
    Separate text features and target labels.
    """

    X = df["text"]
    y = df["label"]

    return X, y


if __name__ == "__main__":
    data = load_sentiment_data()

    print("Dataset loaded successfully.")
    print(f"Shape: {data.shape}")
    print("\nClass distribution:")
    print(data["sentiment"].value_counts())
    print("\nSample:")
    print(data.head())