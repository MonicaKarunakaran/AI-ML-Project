import os
import pandas as pd

from src.pipeline import make_data


def test_make_data():

    df = make_data()

    assert os.path.exists("data/synthetic.csv")

    assert isinstance(df, pd.DataFrame)

    assert df.shape[0] == 500

    assert "target" in df.columns

    assert df.isnull().sum().sum() == 0