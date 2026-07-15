import os

csv_size = os.path.getsize("cleaned_dataset.csv")
parquet_size = os.path.getsize("cleaned_dataset.parquet")

print("\nFILE SIZE COMPARISON")
print("-" * 30)

print(f"CSV File Size     : {csv_size} bytes")
print(f"Parquet File Size : {parquet_size} bytes")

if csv_size > parquet_size:
    print("\nParquet is smaller and more storage efficient.")
elif parquet_size > csv_size:
    print("\nCSV is smaller for this dataset after cleaning.")
else:
    print("\nBoth files have the same size.")