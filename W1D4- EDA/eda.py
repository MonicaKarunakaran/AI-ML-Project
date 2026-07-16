# ============================================================
# W1D4 - Exploratory Data Analysis (EDA)
# Stack: Pandas + Matplotlib + Seaborn
# Dataset: Titanic
# ============================================================

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------------------------------
# Create folder to save plots
# ------------------------------------------------------------
os.makedirs("plots", exist_ok=True)

# ------------------------------------------------------------
# Load Dataset
# ------------------------------------------------------------
df = pd.read_csv("data/train.csv")

# ------------------------------------------------------------
# Basic Dataset Information
# ------------------------------------------------------------
print("=" * 60)
print("FIRST 5 ROWS")
print("=" * 60)
print(df.head())

print("\n" + "=" * 60)
print("DATASET SHAPE")
print("=" * 60)
print(df.shape)

print("\n" + "=" * 60)
print("COLUMN NAMES")
print("=" * 60)
print(df.columns.tolist())

print("\n" + "=" * 60)
print("DESCRIPTIVE STATISTICS")
print("=" * 60)
print(df.describe(include="all"))

print("\n" + "=" * 60)
print("DATASET INFORMATION")
print("=" * 60)
df.info()

print("\n" + "=" * 60)
print("MISSING VALUES")
print("=" * 60)
print(df.isnull().sum())

print("\n" + "=" * 60)
print("DUPLICATE ROWS")
print("=" * 60)
print(df.duplicated().sum())

# ------------------------------------------------------------
# Five Observations
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("FIVE OBSERVATIONS")
print("=" * 60)

print("1. The dataset contains 891 rows and 12 columns.")
print("2. The Cabin column has the highest number of missing values (687).")
print("3. The Age column contains 177 missing values, while Embarked has only 2.")
print("4. The dataset contains both numerical and categorical features.")
print("5. The target column 'Survived' has no missing values and is ready for model training after preprocessing.")

# ------------------------------------------------------------
# Distribution Plots
# ------------------------------------------------------------
numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns

for col in numeric_cols:
    plt.figure(figsize=(7,4))

    sns.histplot(
        df[col].dropna(),
        bins=30,
        kde=True,
        color="steelblue"
    )

    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Count")

    plt.tight_layout()
    plt.savefig(f"plots/{col}_distribution.png")
    plt.show()

# ------------------------------------------------------------
# Correlation Heatmap
# ------------------------------------------------------------
plt.figure(figsize=(10,8))

corr = df[numeric_cols].corr()

sns.heatmap(
    corr,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5,
    square=True
)

plt.title("Correlation Heatmap")

plt.tight_layout()
plt.savefig("plots/correlation_heatmap.png")
plt.show()

# ------------------------------------------------------------
# Top 10 Category Counts
# ------------------------------------------------------------
cat_cols = df.select_dtypes(include=["object", "category"]).columns

for col in cat_cols:

    top10 = df[col].value_counts().head(10)

    plt.figure(figsize=(8,4))

    sns.barplot(
        x=top10.values,
        y=top10.index,
        hue=top10.index,
        palette="viridis",
        legend=False
    )

    plt.title(f"Top 10 Categories - {col}")
    plt.xlabel("Count")
    plt.ylabel(col)

    plt.tight_layout()
    plt.savefig(f"plots/{col}_top10.png")
    plt.show()

# ------------------------------------------------------------
# 200-word EDA Narrative
# ------------------------------------------------------------
rows, cols = df.shape
num_cols = len(numeric_cols)
cat_cols_count = len(cat_cols)

eda_narrative = f"""
============================================================
Exploratory Data Analysis (EDA) Summary
============================================================

The Titanic dataset contains {rows} passenger records and {cols}
features. Among these, {num_cols} are numerical features and
{cat_cols_count} are categorical features. Initial exploration
using describe(), info(), and isnull().sum() provided an overview
of the dataset structure, summary statistics, and data quality.

The analysis identified missing values in the Age, Cabin, and
Embarked columns. Cabin has the largest number of missing values,
indicating that it may require removal or advanced imputation.
Age also contains several missing values that should be filled
using an appropriate strategy such as the median.

Distribution plots of numerical features show the spread of Age,
Fare, Passenger Class, SibSp, and Parch. The Fare distribution is
right-skewed with a few high-value outliers. The correlation
heatmap indicates that most numerical features have weak to
moderate correlations, suggesting that multiple variables
contribute independently to survival prediction.

The categorical plots reveal that passenger class, sex, and
embarkation point are unevenly distributed across the dataset.
Overall, the dataset is suitable for machine learning after
handling missing values, encoding categorical variables,
performing feature engineering, and scaling numerical features
where necessary.
"""

print(eda_narrative)

print("\nEDA Completed Successfully!")