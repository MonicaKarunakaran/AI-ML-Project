import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data():

    from sklearn.datasets import load_iris
    iris = load_iris()
    df = pd.DataFrame(
        iris.data,
        columns=iris.feature_names
    )
    df["target"] = iris.target
    return df

def preprocess_data(df):

    X = df.drop("target", axis=1)
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        scaler
    )