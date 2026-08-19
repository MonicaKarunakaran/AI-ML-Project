import csv
from pathlib import Path

from langchain_core.tools import tool


DATA_FILE = Path(__file__).resolve().parents[2] / "data" / "holidays.csv"


@tool
def date_lookup_tool(date: str) -> str:
    """Look up a holiday for a given date."""

    if not DATA_FILE.exists():
        return f"No holiday data found for {date}."

    with DATA_FILE.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)

        for row in reader:
            if row["date"] == date:
                return f"{row['holiday']} is on {date}."

    return f"No holiday found for {date}."