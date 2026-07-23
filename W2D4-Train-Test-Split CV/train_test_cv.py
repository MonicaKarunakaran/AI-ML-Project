# W2D4 - Train/Test Split & Cross Validation

import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


data = load_breast_cancer()

X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target


print("Dataset Shape:", X.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


print("Training Data:", X_train.shape)
print("Testing Data:", X_test.shape)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000)

model.fit(
    X_train_scaled,
    y_train
)

y_pred = model.predict(X_test_scaled)

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\nTest Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

cv_scores = cross_val_score(
    model,
    X_train_scaled,
    y_train,
    cv=5
)


print("\nCross Validation Scores:")
print(cv_scores)

print(
    "Average CV Accuracy:",
    cv_scores.mean()
)


scalers = {
    "StandardScaler": StandardScaler(),
    "MinMaxScaler": MinMaxScaler(),
    "RobustScaler": RobustScaler()
}


for name, scaler in scalers.items():

    X_scaled = scaler.fit_transform(X)

    scores = cross_val_score(
        model,
        X_scaled,
        y,
        cv=5
    )

    print(
        f"{name} CV Accuracy:",
        scores.mean()
    )