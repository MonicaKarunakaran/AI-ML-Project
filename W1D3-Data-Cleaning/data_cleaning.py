"""
W1D3 - Data Loading, Cleaning & Inspection
Dataset: Indian Startup Funding Dataset

Tasks:
1. Load dataset
2. Inspect data
3. Clean column names
4. Handle missing values
5. Remove duplicates
6. Convert data types
7. Save cleaned dataset
"""

import pandas as pd
import numpy as np


# -----------------------------------
# 1. Load Dataset
# -----------------------------------

df = pd.read_csv("startup_funding.csv")


print("\nOriginal Dataset:")
print(df.head())


# -----------------------------------
# 2. Dataset Inspection
# -----------------------------------

print("\nDataset Shape:")
print(df.shape)


print("\nColumns:")
print(df.columns)


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

# Fill categorical missing values

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


# Remarks has many missing values
df["remarks"] = df["remarks"].fillna("No Remarks")


# Amount column missing values
df["amount_in_usd"] = (
    df["amount_in_usd"]
    .fillna("0")
)


# -----------------------------------
# 5. Clean Amount Column
# -----------------------------------

df["amount_in_usd"] = (
    df["amount_in_usd"]
    .str.replace(",", "")
)


df["amount_in_usd"] = pd.to_numeric(
    df["amount_in_usd"],
    errors="coerce"
)


# Replace remaining NaN values
df["amount_in_usd"] = df["amount_in_usd"].fillna(0)


# -----------------------------------
# 6. Remove Duplicate Rows
# -----------------------------------

duplicates = df.duplicated().sum()

print("\nDuplicate Records:")
print(duplicates)


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
    "cleaned_startup_funding.csv",
    index=False
)


print("\nCleaning Completed Successfully!")