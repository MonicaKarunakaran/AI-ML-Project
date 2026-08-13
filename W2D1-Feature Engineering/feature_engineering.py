import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import (
    LabelEncoder,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
    MinMaxScaler,
    RobustScaler
)

from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_classif

# -------------------------
# Load Dataset
# -------------------------
df = pd.read_csv("data/train.csv")

print("Dataset Shape:", df.shape)
print()

# -------------------------
# Fill Missing Values
# -------------------------
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# -------------------------
# LABEL ENCODER
# -------------------------
le = LabelEncoder()
df["Sex_Label"] = le.fit_transform(df["Sex"])

print("Label Encoding:")
print(df[["Sex", "Sex_Label"]].head())
print()

# -------------------------
# ORDINAL ENCODER
# -------------------------
ordinal = OrdinalEncoder(categories=[["S", "C", "Q"]])

df["Embarked_Ordinal"] = ordinal.fit_transform(df[["Embarked"]])

print("Ordinal Encoding:")
print(df[["Embarked", "Embarked_Ordinal"]].head())
print()

# -------------------------
# ONE HOT ENCODER
# -------------------------
encoder = ColumnTransformer(
    transformers=[
        ("onehot", OneHotEncoder(drop="first"), ["Embarked"])
    ],
    remainder="passthrough"
)

encoded = encoder.fit_transform(df)

print("OneHot Encoding Completed")
print()

# -------------------------
# Scaling
# -------------------------

scalers = {
    "StandardScaler": StandardScaler(),
    "MinMaxScaler": MinMaxScaler(),
    "RobustScaler": RobustScaler()
}

for column in ["Age", "Fare"]:

    plt.figure(figsize=(10,4))

    plt.subplot(1,4,1)
    plt.hist(df[column], bins=20)
    plt.title("Original")

    i = 2

    for name, scaler in scalers.items():

        scaled = scaler.fit_transform(df[[column]])

        plt.subplot(1,4,i)
        plt.hist(scaled, bins=20)
        plt.title(name)

        i += 1

    plt.tight_layout()
    plt.savefig(f"plots/{column.lower()}_before_after_scaling.png")
    plt.close()

print("Scaling Completed")
print()

# -------------------------
# Feature Selection
# -------------------------

features = df[[
    "Pclass",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "Sex_Label",
    "Embarked_Ordinal"
]]

target = df["Survived"]

selector = SelectKBest(score_func=f_classif, k=5)

selected = selector.fit(features, target)

scores = pd.DataFrame({
    "Feature": features.columns,
    "Score": selected.scores_
})

scores = scores.sort_values("Score", ascending=False)

print("Top 5 Features")
print(scores.head())

print("\nCompleted Successfully.")