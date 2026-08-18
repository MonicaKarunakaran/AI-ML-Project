import pandas as pd

INPUT_FILE = "token_cost_audit.csv"
OUTPUT_FILE = "token_cost_audit.xlsx"

# Read audit results
df = pd.read_csv(INPUT_FILE)

# Create Excel workbook
with pd.ExcelWriter(
    OUTPUT_FILE,
    engine="openpyxl",
) as writer:

    # Main audit
    df.to_excel(
        writer,
        sheet_name="Token Audit",
        index=False,
    )

    # Summary
    summary = pd.DataFrame(
        {
            "Metric": [
                "Number of prompts",
                "Total input tokens",
                "Total estimated output tokens",
                "Total GPT-4o cost (USD)",
                "Total GPT-4o cost (INR)",
                "Total GPT-4o Mini cost (USD)",
                "Total GPT-4o Mini cost (INR)",
            ],
            "Value": [
                len(df),
                df["Input Tokens"].sum(),
                df["Estimated Output Tokens"].sum(),
                df["GPT-4o Cost (USD)"].sum(),
                df["GPT-4o Cost (INR)"].sum(),
                df["GPT-4o Mini Cost (USD)"].sum(),
                df["GPT-4o Mini Cost (INR)"].sum(),
            ],
        }
    )

    summary.to_excel(
        writer,
        sheet_name="Summary",
        index=False,
    )

print(
    f"Excel audit created successfully: {OUTPUT_FILE}"
)