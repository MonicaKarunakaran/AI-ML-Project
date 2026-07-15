import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    OrdinalEncoder,
    RobustScaler,
    StandardScaler,
)

# -------------------------------------------------------------------------
# 1. SETUP & DATA PREPARATION
# -------------------------------------------------------------------------
INPUT_PATH = Path("data/startup_funding.csv")
if not INPUT_PATH.exists():
    raise FileNotFoundError(f"Missing required file: {INPUT_PATH}")

df = pd.read_csv(INPUT_PATH)

# Clean Target Column: Convert "Amount in USD" to numeric values
df["Amount_USD"] = (
    df["Amount in USD"]
    .astype(str)
    .str.replace(",", "", regex=False)
    .str.replace(r"[+xX\s]", "", regex=True)
)
df["Amount_USD"] = pd.to_numeric(df["Amount_USD"], errors="coerce")
df = df.dropna(subset=["Amount_USD"])  # Drop records missing target labels

# Engineer Base Numerical and Categorical Columns
df["Startup_Name_Len"] = df["Startup Name"].astype(str).apply(len)
df["Num_Investors"] = (
    df["Investors Name"]
    .astype(str)
    .apply(lambda x: len(x.split(",")) if pd.notna(x) else 1)
)

# Standardize values to reduce structural noise
df["City"] = df["City  Location"].astype(str).str.strip().fillna("Unknown")
df["Investment_Type"] = (
    df["InvestmentnType"].astype(str).str.strip().fillna("Unknown")
)

# -------------------------------------------------------------------------
# 2. TASK 2.1: CATEGORICAL ENCODING
# -------------------------------------------------------------------------
print("\n=== CATEGORICAL ENCODING TRADEOFFS ===")

# LabelEncoder (Typically used on target variables)
le = LabelEncoder()
df["City_LabelEncoded"] = le.fit_transform(df["City"])

# OrdinalEncoder (Best for ordinal features with clear intrinsic ranking)
ordinal_cats = [["Unknown", "Seed", "Series A", "Series B", "Series C", "Series D"]]
# Create a dummy rank column for representation
df["Funding_Stage_Rank"] = df["Investment_Type"].apply(
    lambda x: (
        x
        if x in ["Seed", "Series A", "Series B", "Series C", "Series D"]
        else "Unknown"
    )
)
oe = OrdinalEncoder(categories=ordinal_cats, handle_unknown="use_encoded_value", unknown_value=-1)
df["Stage_OrdinalEncoded"] = oe.fit_transform(df[["Funding_Stage_Rank"]])

# OneHotEncoder (Best for nominal variables without intrinsic sequence)
# Group low-frequency categories to prevent massive dimensionality explosion
top_cities = df["City"].value_counts().index[:8]
df["City_Grouped"] = df["City"].apply(lambda x: x if x in top_cities else "Other")
df_ohe = pd.get_dummies(df, columns=["City_Grouped"], prefix="City_OHE", drop_first=True)

print("Categorical variables successfully encoded.")

# -------------------------------------------------------------------------
# 3. TASK 2.2: FEATURE SCALING & VISUALIZATION
# -------------------------------------------------------------------------
print("\n=== FEATURE SCALING & PLOTTING ===")

# Extract target and skewed numerical feature for comparison
# Note: Indian startup funding has massive outlier skew (e.g., Flipkart/Paytm mega rounds)
raw_amounts = df[["Amount_USD"]].copy()

# Apply different Scalers
std_scaler = StandardScaler()
minmax_scaler = MinMaxScaler()
robust_scaler = RobustScaler()

df_scaled = pd.DataFrame(
    {
        "Raw (Amount USD)": raw_amounts["Amount_USD"],
        "StandardScaler": std_scaler.fit_transform(raw_amounts).flatten(),
        "MinMaxScaler": minmax_scaler.fit_transform(raw_amounts).flatten(),
        "RobustScaler": robust_scaler.fit_transform(raw_amounts).flatten(),
    }
)

# Plotting the distributions before/after scaling
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Scaling Distribution Transformation Comparison", fontsize=16)

# Raw Distribution (Skewed)
sns.histplot(df_scaled["Raw (Amount USD)"], ax=axes[0, 0], kde=True, color="blue")
axes[0, 0].set_title("Before Scaling (Highly Skewed)")

# StandardScaler
sns.histplot(df_scaled["StandardScaler"], ax=axes[0, 1], kde=True, color="orange")
axes[0, 1].set_title("After Standard Scaling (mu=0, sigma=1)")

# MinMaxScaler
sns.histplot(df_scaled["MinMaxScaler"], ax=axes[1, 0], kde=True, color="green")
axes[1, 0].set_title("After Min-Max Scaling (Bounds [0, 1])")

# RobustScaler
sns.histplot(df_scaled["RobustScaler"], ax=axes[1, 1], kde=True, color="purple")
axes[1, 1].set_title("After Robust Scaling (Uses Median & IQR)")

plt.tight_layout()
plot_path = "scaling_distributions.png"
plt.savefig(plot_path)
print(f"Distribution comparison plot saved to: {plot_path}")

# -------------------------------------------------------------------------
# 4. TASK 2.3: FEATURE SELECTION (SELECTKBEST)
# -------------------------------------------------------------------------
print("\n=== FEATURE SELECTION USING SELECTKBEST ===")

# Assemble all numeric features for selection
feature_cols = [
    "Startup_Name_Len",
    "Num_Investors",
    "City_LabelEncoded",
    "Stage_OrdinalEncoded",
]

# Add dummy columns from OHE
ohe_cols = [col for col in df_ohe.columns if "City_OHE" in col]
X = df_ohe[feature_cols + ohe_cols].fillna(0)
y = df_ohe["Amount_USD"]

# Select top 5 features using f_regression (since target variable is continuous)
selector = SelectKBest(score_func=f_regression, k=5)
X_new = selector.fit_transform(X, y)

selected_features = X.columns[selector.get_support()]
scores = selector.scores_[selector.get_support()]

print("Top 5 Identified Features with selectKBest Scores:")
for feat, score in zip(selected_features, scores):
    print(f"- {feat:<25} Score: {score:.4f}")