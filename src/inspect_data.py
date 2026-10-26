
from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "data" / "college_emails.csv"
OUTPUT_DIR = PROJECT_DIR / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}\n"
        "Place college_emails.csv inside the data folder."
    )

# Read the dataset
df = pd.read_csv(DATA_PATH, low_memory=False)

print("\n--- DATASET OVERVIEW ---")
print("Rows:", len(df))
print("Columns:", df.columns.tolist())

print("\n--- DATA TYPES ---")
print(df.dtypes)

print("\n--- MISSING VALUES ---")
print(df.isna().sum())

print("\n--- EXACT DUPLICATE ROWS ---")
print(df.duplicated().sum())

print("\n--- CATEGORY DISTRIBUTION ---")
if "category" in df.columns:
    print(df["category"].fillna("(blank)").value_counts().to_string())

print("\n--- SAMPLE SUBJECTS ---")
if "subject" in df.columns:
    print(df["subject"].head(15).to_string(index=False))

print("\n--- SAMPLE EMAILS ---")
for _, row in df.head(5).iterrows():
    print("\nSubject:", row.get("subject", ""))
    body = str(row.get("body", ""))
    print("Body:", body[:300].replace("\n", " "))

# Save a report without modifying the original
report = pd.DataFrame({
    "column": df.columns,
    "missing_values": [df[col].isna().sum() for col in df.columns],
    "missing_percent": [
        round(df[col].isna().mean() * 100, 2)
        for col in df.columns
    ],
    "data_type": [str(df[col].dtype) for col in df.columns],
})

report.to_csv(OUTPUT_DIR / "data_quality_report.csv", index=False)
print("\nSaved: outputs/data_quality_report.csv")
