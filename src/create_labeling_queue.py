
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = ROOT / "outputs" / "college_emails_clean.csv"
OUTPUT_FILE = ROOT / "outputs" / "labeling_queue.csv"

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Cleaned dataset not found: {INPUT_FILE}"
    )

df = pd.read_csv(
    INPUT_FILE,
    keep_default_na=False,
    low_memory=False
)

# Keep only emails with some text.
df = df[
    df["subject"].str.strip().ne("")
    | df["body"].str.strip().ne("")
].copy()

# Create a blank category column if it does not exist.
if "category" not in df.columns:
    df["category"] = ""

# Keep the existing labels, if any.
df["category"] = df["category"].str.strip().str.lower()

# Save a separate file for manual labeling.
df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

print("Labeling queue created successfully.")
print("Total emails:", len(df))
print("Already labeled:", df["category"].ne("").sum())
print("Still unlabeled:", df["category"].eq("").sum())
print("Saved to:", OUTPUT_FILE)
