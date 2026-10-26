
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

ORIGINAL = ROOT / "data" / "college_emails.csv"
CLEANED = ROOT / "outputs" / "college_emails_clean.csv"

if not ORIGINAL.exists() or not CLEANED.exists():
    raise FileNotFoundError(
        "Check that the original and cleaned CSV files exist."
    )

# Read both files with the same CSV parser.
original = pd.read_csv(ORIGINAL, keep_default_na=False, low_memory=False)
cleaned = pd.read_csv(CLEANED, keep_default_na=False, low_memory=False)

print("Original rows:", len(original))
print("Cleaned rows:", len(cleaned))

for col in ["subject", "body"]:
    if col not in original.columns or col not in cleaned.columns:
        raise ValueError(f"Missing required column: {col}")

    original_lengths = original[col].astype(str).str.len()
    cleaned_lengths = cleaned[col].astype(str).str.len()

    print(f"\n{col.upper()} LENGTHS")
    print("Original maximum:", original_lengths.max())
    print("Cleaned maximum:", cleaned_lengths.max())

# Match emails using subject and received_at where available.
keys = ["subject"]
if "received_at" in original.columns and "received_at" in cleaned.columns:
    keys.append("received_at")

for frame in [original, cleaned]:
    for key in keys:
        frame[key] = frame[key].astype(str).str.strip()

# Build a lookup of cleaned bodies by email key.
lookup = {}
for _, row in cleaned.iterrows():
    key = tuple(row[k] for k in keys)
    lookup.setdefault(key, []).append(str(row["body"]))

matched = 0
different = 0
missing = 0

for _, row in original.iterrows():
    key = tuple(str(row[k]).strip() for k in keys)
    candidates = lookup.get(key)

    if not candidates:
        missing += 1
        continue

    body = str(row["body"]).strip()

    if body in candidates:
        matched += 1
        candidates.remove(body)
    else:
        # Check whether the body is a prefix of a cleaned body
        # or vice versa; report only counts, not email contents.
        if any(body.startswith(c) or c.startswith(body) for c in candidates):
            different += 1
        else:
            different += 1

print("\nBODY INTEGRITY RESULTS")
print("Exact body matches after trimming whitespace:", matched)
print("Matched keys but different bodies:", different)
print("Original rows with no matching key:", missing)
print("\nNote: duplicate subjects can make key matching ambiguous.")
