# ==========================================================
# W2D3: Handling Imbalanced Data using SMOTE
# AI/ML Internship - Cynaris Solutions
# ==========================================================

# Import required libraries
import os
from collections import Counter

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from imblearn.over_sampling import SMOTE

OUTPUT_FOLDER = "outputs"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

print("=" * 60)
print("Loading Breast Cancer Dataset")
print("=" * 60)

data = load_breast_cancer()

X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name="target")

print("\nDataset Shape:")
print(X.shape)

print("\nFirst Five Rows:")
print(X.head())

print("\nMissing Values:")
print(X.isnull().sum().sum())

print("\nClass Distribution:")
print(y.value_counts())

plt.figure(figsize=(6, 4))
sns.countplot(x=y)
plt.title("Class Distribution Before SMOTE")
plt.xlabel("Class")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FOLDER, "before_smote.png"))
plt.show()

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

print("\n" + "=" * 60)
print("MODEL BEFORE SMOTE")
print("=" * 60)

model = LogisticRegression(max_iter=5000)

model.fit(X_train, y_train)

prediction_before = model.predict(X_test)

accuracy_before = accuracy_score(y_test, prediction_before)

print(f"\nAccuracy Before SMOTE: {accuracy_before:.4f}")

print("\nClassification Report Before SMOTE")

print(classification_report(y_test, prediction_before))

print("=" * 60)
print("Applying SMOTE")
print("=" * 60)

print("\nBefore SMOTE")

print(Counter(y_train))

smote = SMOTE(random_state=42)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train,
    y_train,
)

print("\nAfter SMOTE")

print(Counter(y_train_smote))

plt.figure(figsize=(6, 4))
sns.countplot(x=y_train_smote)
plt.title("Class Distribution After SMOTE")
plt.xlabel("Class")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FOLDER, "after_smote.png"))
plt.show()

print("\n" + "=" * 60)
print("MODEL AFTER SMOTE")
print("=" * 60)

model_smote = LogisticRegression(max_iter=5000)

model_smote.fit(
    X_train_smote,
    y_train_smote,
)

prediction_after = model_smote.predict(X_test)

accuracy_after = accuracy_score(
    y_test,
    prediction_after,
)

print(f"\nAccuracy After SMOTE: {accuracy_after:.4f}")

print("\nClassification Report After SMOTE")

print(classification_report(
    y_test,
    prediction_after,
))

cm = confusion_matrix(
    y_test,
    prediction_after,
)

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
)

plt.title("Confusion Matrix After SMOTE")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_FOLDER,
        "confusion_matrix.png",
    )
)

plt.show()

print("\n" + "=" * 60)
print("COMPARISON")
print("=" * 60)

print(f"Accuracy Before SMOTE : {accuracy_before:.4f}")
print(f"Accuracy After SMOTE  : {accuracy_after:.4f}")

if accuracy_after >= accuracy_before:
    print("\nModel performance improved or remained stable after SMOTE.")
else:
    print("\nModel performance decreased after SMOTE.")

print("\nTask Completed Successfully!")

print("\nOutput images saved in:")
print(OUTPUT_FOLDER)