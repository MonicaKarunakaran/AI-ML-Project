"""
W1D3 - Data Loading, Cleaning & Inspection

Dataset:
Indian Startup Funding Dataset

Tasks:
1. Load dataset
2. Inspect data
3. Clean column names
4. Handle missing values
5. Remove duplicates
6. Convert investment amount to numeric
7. Perform NumPy operations
8. Save cleaned dataset
"""

import pandas as pd
import numpy as np


# -----------------------------------
# 1. Load Dataset
# -----------------------------------

df = pd.read_csv("data/startup_funding.csv")


print("\nOriginal Dataset:")
print(df.head())


# -----------------------------------
# 2. Dataset Inspection
# -----------------------------------

print("\nDataset Shape:")
print(df.shape)


print("\nColumns:")
print(df.columns)


print("\nDataset Information:")
print(df.info())


print("\nMissing Values:")
print(df.isnull().sum())


# -----------------------------------
# 3. Clean Column Names
# -----------------------------------

df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
)


print("\nCleaned Column Names:")
print(df.columns)


# -----------------------------------
# 4. Handle Missing Values
# -----------------------------------

# Fill missing categorical values

categorical_columns = [
    "industry_vertical",
    "subvertical",
    "city__location",
    "investors_name",
    "investmentntype"
]


for column in categorical_columns:
    if column in df.columns:
        df[column] = df[column].fillna("Unknown")


# Fill missing remarks

df["remarks"] = df["remarks"].fillna("No Remarks")


# Fill missing investment values

df["amount_in_usd"] = df["amount_in_usd"].fillna("0")


# -----------------------------------
# 5. Convert Investment Amount
# -----------------------------------

# Remove commas from amount

df["amount_in_usd"] = (
    df["amount_in_usd"]
    .str.replace(",", "")
)


# Convert string to numeric

df["amount_in_usd"] = pd.to_numeric(
    df["amount_in_usd"],
    errors="coerce"
)


# Replace invalid values

df["amount_in_usd"] = (
    df["amount_in_usd"]
    .fillna(0)
)


# -----------------------------------
# 6. Remove Duplicate Records
# -----------------------------------

duplicate_count = df.duplicated().sum()

print("\nDuplicate Records:")
print(duplicate_count)


df = df.drop_duplicates()


# -----------------------------------
# 7. NumPy Operation
# -----------------------------------

investment_array = np.array(
    df["amount_in_usd"]
)


print("\nInvestment Array Sample:")
print(investment_array[:10])


print("\nAverage Investment:")
print(np.mean(investment_array))


# -----------------------------------
# 8. Save Clean Dataset
# -----------------------------------

df.to_csv(
    "data/cleaned_startup_funding.csv",
    index=False
)


print("\nCleaning Completed Successfully!")

print("\nCleaned Dataset Saved:")
print("data/cleaned_startup_funding.csv")

# Final pipeline submission verification