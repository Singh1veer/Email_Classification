
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = ROOT / "outputs" / "auto_categorized_emails.csv"
OUTPUT_FILE = ROOT / "outputs" / "validation_sample_50.csv"

df = pd.read_csv(
    INPUT_FILE,
    keep_default_na=False,
    low_memory=False
)

# Sample from both groups.
needs_review = df[df["review_required"].astype(str).str.lower() == "true"]
not_flagged = df[df["review_required"].astype(str).str.lower() == "false"]

review_sample = needs_review.sample(
    n=min(25, len(needs_review)),
    random_state=42
)

other_sample = not_flagged.sample(
    n=min(25, len(not_flagged)),
    random_state=42
)

sample = pd.concat(
    [review_sample, other_sample],
    ignore_index=True
).sample(frac=1, random_state=42)

# Keep the predicted label and provide a separate field for human judgment.
sample = sample[
    ["subject", "body", "predicted_category", "matched_categories",
     "review_required"]
].copy()

sample["verified_category"] = ""
sample["review_notes"] = ""

sample.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)

print("Validation sample created.")
print("Emails selected:", len(sample))
print("Flagged for review:", sample["review_required"].astype(str).str.lower().eq("true").sum())
print("File:", OUTPUT_FILE)
