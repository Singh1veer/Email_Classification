
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = ROOT / "outputs" / "labeling_queue.csv"
OUTPUT_FILE = ROOT / "outputs" / "sample_100_emails.csv"

df = pd.read_csv(
    INPUT_FILE,
    keep_default_na=False,
    low_memory=False
)

sample = df.sample(
    n=min(100, len(df)),
    random_state=42
).copy()

# Keep useful columns for manual labeling.
sample = sample[["subject", "body", "category"]]

# Add an empty category column if necessary.
sample["category"] = ""

sample.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)

print("Sample created successfully.")
print("Emails to label:", len(sample))
print("File:", OUTPUT_FILE)
