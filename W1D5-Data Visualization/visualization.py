import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Style
sns.set(style="whitegrid")

# Create plots folder
os.makedirs("plots", exist_ok=True)

# Load dataset
df = pd.read_csv("data/train.csv")

print(df.head())
print(df.info())

# ------------------------------------
# 1. Age Distribution
# ------------------------------------

plt.figure(figsize=(8,5))
sns.histplot(df["Age"].dropna(), bins=30, kde=True)

plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")

plt.savefig("plots/age_distribution.png")
plt.show()

# ------------------------------------
# 2. Survival Count
# ------------------------------------

plt.figure(figsize=(6,5))
sns.countplot(x="Survived", data=df)

plt.title("Passenger Survival")
plt.xlabel("Survived")
plt.ylabel("Count")

plt.savefig("plots/survival_count.png")
plt.show()

# ------------------------------------
# 3. Survival by Gender
# ------------------------------------

plt.figure(figsize=(6,5))
sns.countplot(x="Sex", hue="Survived", data=df)

plt.title("Survival by Gender")

plt.savefig("plots/survival_by_gender.png")
plt.show()

# ------------------------------------
# 4. Fare Boxplot
# ------------------------------------

plt.figure(figsize=(8,5))
sns.boxplot(x=df["Fare"])

plt.title("Fare Distribution")

plt.savefig("plots/fare_boxplot.png")
plt.show()

# ------------------------------------
# 5. Correlation Heatmap
# ------------------------------------

numeric_df = df.select_dtypes(include=["number"])

plt.figure(figsize=(10,8))
sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap")

plt.savefig("plots/correlation_heatmap.png")
plt.show()

# ------------------------------------
# 6. Pairplot
# ------------------------------------

pairplot = sns.pairplot(
    df[["Age","Fare","Pclass","Survived"]].dropna(),
    hue="Survived"
)

pairplot.savefig("plots/pairplot.png")

print("\nAll plots generated successfully!")