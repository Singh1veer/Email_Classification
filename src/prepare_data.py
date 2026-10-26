
from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "data" / "college_emails.csv"
OUTPUT_DIR = PROJECT_DIR / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}\n"
        "Place college_emails.csv inside the data folder."
    )

# Read the original dataset
df = pd.read_csv(DATA_PATH, low_memory=False)

required = {"subject", "body", "category"}
missing = required - set(df.columns)

if missing:
    raise ValueError(f"Missing required columns: {missing}")

# Clean missing text values
df["subject"] = df["subject"].fillna("").astype(str).str.strip()
df["body"] = df["body"].fillna("").astype(str).str.strip()

# Remove rows with no usable email text
df = df[
    (df["subject"] != "") | (df["body"] != "")
].copy()

# Remove exact duplicate rows
df = df.drop_duplicates().reset_index(drop=True)

# Preserve existing category labels, if any
df["category"] = df["category"].fillna("").astype(str).str.strip()

# Save the cleaned working copy
clean_path = OUTPUT_DIR / "college_emails_clean.csv"
df.to_csv(clean_path, index=False)

# Prepare a sample for manual labeling
sample_size = min(500, len(df))
queue = df.sample(n=sample_size, random_state=42).copy()

# Leave categories blank for manual review
queue["category"] = ""

queue_path = OUTPUT_DIR / "labeling_queue.csv"
queue.to_csv(queue_path, index=False)

print("Data preparation completed successfully!")
print("Original rows:", len(pd.read_csv(DATA_PATH, usecols=["subject"])))
print("Cleaned rows:", len(df))
print("Rows in labeling queue:", len(queue))
print("Saved cleaned data to:", clean_path)
print("Saved labeling queue to:", queue_path)
