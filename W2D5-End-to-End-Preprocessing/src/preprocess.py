import os
import joblib
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder
)


def create_pipeline():

    numeric_features = [
        "Pclass",
        "Age",
        "SibSp",
        "Parch",
        "Fare"
    ]

    categorical_features = [
        "Sex",
        "Embarked"
    ]


    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            ),

            (
                "scaler",
                StandardScaler()
            )
        ]
    )


    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                )
            ),

            (
                "encoder",
                OneHotEncoder(
                    drop="first",
                    handle_unknown="ignore",
                    sparse_output=False
                )
            )
        ]
    )


    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                numeric_pipeline,
                numeric_features
            ),

            (
                "cat",
                categorical_pipeline,
                categorical_features
            )
        ]
    )


    return preprocessor



def preprocess_data(df):

    print("Starting preprocessing...")


    X = df.drop(
        "Survived",
        axis=1
    )

    y = df["Survived"]


    pipeline = create_pipeline()


    X_processed = pipeline.fit_transform(X)


    encoded_columns = (
        pipeline
        .named_transformers_["cat"]
        .named_steps["encoder"]
        .get_feature_names_out(
            [
                "Sex",
                "Embarked"
            ]
        )
    )


    columns = [
        "Pclass",
        "Age",
        "SibSp",
        "Parch",
        "Fare"
    ] + list(encoded_columns)



    processed_df = pd.DataFrame(
        X_processed,
        columns=columns
    )


    processed_df["Survived"] = y.values


    os.makedirs(
        "output",
        exist_ok=True
    )


    processed_df.to_csv(
        "output/titanic_preprocessed.csv",
        index=False
    )


    joblib.dump(
        pipeline,
        "output/preprocessing_pipeline.pkl"
    )


    print("Preprocessing completed")

    return processed_df