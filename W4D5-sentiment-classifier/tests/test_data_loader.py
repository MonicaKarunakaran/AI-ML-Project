"""
Tests for data loading.
"""

from pathlib import Path

import pandas as pd
import pytest

from src.data_loader import (
    get_features_and_target,
    load_sentiment_data,
)


def test_load_sentiment_data(tmp_path):

    data = pd.DataFrame(
        {
            "text": [
                "I love this",
                "This is terrible",
                "Amazing product",
                "Very bad experience",
            ],
            "sentiment": [
                "positive",
                "negative",
                "positive",
                "negative",
            ],
        }
    )

    file_path = tmp_path / "sentiment.csv"

    data.to_csv(
        file_path,
        index=False,
    )

    result = load_sentiment_data(
        file_path
    )

    assert len(result) == 4
    assert "text" in result.columns
    assert "sentiment" in result.columns
    assert "label" in result.columns


def test_missing_columns(tmp_path):

    data = pd.DataFrame(
        {
            "review": [
                "Good product"
            ],
            "sentiment": [
                "positive"
            ],
        }
    )

    file_path = tmp_path / "invalid.csv"

    data.to_csv(
        file_path,
        index=False,
    )

    with pytest.raises(ValueError):

        load_sentiment_data(
            file_path
        )


def test_get_features_and_target():

    data = pd.DataFrame(
        {
            "text": [
                "Good",
                "Bad",
            ],
            "sentiment": [
                "positive",
                "negative",
            ],
            "label": [
                1,
                0,
            ],
        }
    )

    X, y = get_features_and_target(
        data
    )

    assert len(X) == 2
    assert len(y) == 2