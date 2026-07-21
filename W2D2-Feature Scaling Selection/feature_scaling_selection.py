import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import (
    LabelEncoder,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
)
from sklearn.feature_selection import SelectKBest, f_classif
df = pd.read_csv("data/train.csv")

print("=" * 60)
print("DATASET PREVIEW")
print("=" * 60)
print(df.head())

data = df[["Survived", "Pclass", "Sex", "Age", "Fare", "Embarked"]].copy()

data["Age"] = data["Age"].fillna(data["Age"].median())
data["Embarked"] = data["Embarked"].fillna(data["Embarked"].mode()[0])

print("\nMissing Values")
print(data.isnull().sum())

print("\n" + "=" * 60)
print("LABEL ENCODER")
print("=" * 60)

label_encoder = LabelEncoder()

data["Sex_LabelEncoded"] = label_encoder.fit_transform(data["Sex"])

print(data[["Sex", "Sex_LabelEncoded"]].head())

print("\n" + "=" * 60)
print("ONE HOT ENCODER")
print("=" * 60)

onehot = OneHotEncoder(sparse_output=False)

encoded = onehot.fit_transform(data[["Embarked"]])

onehot_df = pd.DataFrame(
    encoded,
    columns=onehot.get_feature_names_out(["Embarked"])
)

print(onehot_df.head())

print("\n" + "=" * 60)
print("ORDINAL ENCODER")
print("=" * 60)

education = pd.DataFrame({
    "Education": [
        "Low",
        "Medium",
        "High",
        "Medium",
        "Low"
    ]
})

ordinal = OrdinalEncoder(
    categories=[["Low", "Medium", "High"]]
)

education["Encoded"] = ordinal.fit_transform(
    education[["Education"]]
)

print(education)

X = data[["Age", "Fare"]]

plt.figure(figsize=(6,4))
plt.hist(X["Fare"], bins=30)
plt.title("Fare Distribution Before Scaling")
plt.xlabel("Fare")
plt.ylabel("Frequency")
plt.savefig("before_scaling.png")
plt.show()

standard_scaler = StandardScaler()
X_standard = standard_scaler.fit_transform(X)

minmax_scaler = MinMaxScaler()
X_minmax = minmax_scaler.fit_transform(X)

robust_scaler = RobustScaler()
X_robust = robust_scaler.fit_transform(X)

print("\n" + "=" * 60)
print("STANDARD SCALER")
print("=" * 60)
print(X_standard[:5])

print("\n" + "=" * 60)
print("MINMAX SCALER")
print("=" * 60)
print(X_minmax[:5])

print("\n" + "=" * 60)
print("ROBUST SCALER")
print("=" * 60)
print(X_robust[:5])

plt.figure(figsize=(6,4))
plt.hist(X_standard[:,1], bins=30)
plt.title("Fare Distribution After Standard Scaling")
plt.xlabel("Scaled Fare")
plt.ylabel("Frequency")
plt.savefig("after_scaling.png")
plt.show()

print("\n" + "=" * 60)
print("SELECT KBEST FEATURE SELECTION")
print("=" * 60)

X_features = data[
    [
        "Pclass",
        "Age",
        "Fare",
        "Sex_LabelEncoded"
    ]
]

y = data["Survived"]

selector = SelectKBest(
    score_func=f_classif,
    k="all"
)

selector.fit(X_features, y)

scores = pd.DataFrame({
    "Feature": X_features.columns,
    "Score": selector.scores_
})

scores = scores.sort_values(
    by="Score",
    ascending=False
)

print(scores)

print("\nTop Features")
print(scores.head(5))


print("\n" + "=" * 60)
print("TASK SUMMARY")
print("=" * 60)

print("✔ Label Encoding completed")
print("✔ One Hot Encoding completed")
print("✔ Ordinal Encoding completed")
print("✔ Standard Scaling completed")
print("✔ MinMax Scaling completed")
print("✔ Robust Scaling completed")
print("✔ Feature Selection completed")
print("✔ before_scaling.png generated")
print("✔ after_scaling.png generated")

print("\nW2D2 Task Completed Successfully!")