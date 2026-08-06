from app.schemas import PredictRequest
from app.crew.preprocess import preprocess


def test_preprocess():

    sample = PredictRequest(
        sepal_length=5.1,
        sepal_width=3.5,
        petal_length=1.4,
        petal_width=0.2
    )

    df = preprocess(sample)

    assert df.shape == (1,4)