import pandas as pd

# ---------------------------------------------------
# Load the dataset
# ---------------------------------------------------
df = pd.read_csv("data/startup_funding.csv")

# ---------------------------------------------------
# Display dataset information
# ---------------------------------------------------
print("=" * 50)
print("DATASET INFORMATION")
print("=" * 50)

print("\nDataset Shape:")
print(df.shape)

print("\nData Types:")
print(df.dtypes)

print("\nFirst 10 Rows:")
print(df.head(10))

# ---------------------------------------------------
# Clean 'Amount in USD'
# Remove commas and convert to numeric
# ---------------------------------------------------
df["Amount in USD"] = (
    df["Amount in USD"]
    .astype(str)
    .str.replace(",", "", regex=False)
)

df["Amount in USD"] = pd.to_numeric(
    df["Amount in USD"],
    errors="coerce"
)

# ---------------------------------------------------
# Filter
# Show startups with funding greater than 50 million USD
# ---------------------------------------------------
print("\n" + "=" * 50)
print("FILTER OPERATION")
print("=" * 50)

filtered_df = df[df["Amount in USD"] > 50000000]

print(filtered_df[
    ["Startup Name", "Industry Vertical", "Amount in USD"]
].head(10))

# ---------------------------------------------------
# GroupBy
# Total funding by Industry
# ---------------------------------------------------
print("\n" + "=" * 50)
print("GROUPBY OPERATION")
print("=" * 50)

grouped_df = (
    df.groupby("Industry Vertical")["Amount in USD"]
    .sum()
    .sort_values(ascending=False)
)

print(grouped_df.head(10))

# ---------------------------------------------------
# Merge
# Merge with sample city-state dataframe
# ---------------------------------------------------
print("\n" + "=" * 50)
print("MERGE OPERATION")
print("=" * 50)

city_df = pd.DataFrame({
    "City  Location": [
        "Bengaluru",
        "Mumbai",
        "New Delhi",
        "Chennai",
        "Hyderabad"
    ],
    "State": [
        "Karnataka",
        "Maharashtra",
        "Delhi",
        "Tamil Nadu",
        "Telangana"
    ]
})

merged_df = pd.merge(
    df,
    city_df,
    on="City  Location",
    how="left"
)

print(merged_df.head(10))

# ---------------------------------------------------
# Pivot Table
# Average funding by Investment Type
# ---------------------------------------------------
print("\n" + "=" * 50)
print("PIVOT TABLE")
print("=" * 50)

pivot_df = pd.pivot_table(
    df,
    values="Amount in USD",
    index="InvestmentnType",
    aggfunc="mean"
)

print(pivot_df)

# ---------------------------------------------------
# Clean dataset
# Remove rows where Startup Name or Amount is missing
# ---------------------------------------------------
cleaned_df = df.dropna(
    subset=["Startup Name", "Amount in USD"]
)

# ---------------------------------------------------
# Export cleaned dataset to CSV
# ---------------------------------------------------
cleaned_df.to_csv(
    "cleaned_dataset.csv",
    index=False
)

# ---------------------------------------------------
# Export cleaned dataset to Parquet
# ---------------------------------------------------
cleaned_df.to_parquet(
    "cleaned_dataset.parquet",
    index=False
)

print("\nCleaned dataset exported successfully!")

print("\nFiles Created:")
print("- cleaned_dataset.csv")
print("- cleaned_dataset.parquet")