import pandas as pd


def validate_input(data):

    if data.sepal_length <= 0:
        raise ValueError("Sepal length must be greater than 0")

    if data.sepal_width <= 0:
        raise ValueError("Sepal width must be greater than 0")

    if data.petal_length <= 0:
        raise ValueError("Petal length must be greater than 0")

    if data.petal_width <= 0:
        raise ValueError("Petal width must be greater than 0")


def preprocess(data):

    validate_input(data)

    df = pd.DataFrame(
        [[
            data.sepal_length,
            data.sepal_width,
            data.petal_length,
            data.petal_width
        ]],
        columns=[
            "sepal_length",
            "sepal_width",
            "petal_length",
            "petal_width"
        ]
    )

    return df