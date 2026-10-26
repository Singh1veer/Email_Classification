
import pandas as pd

path = "outputs/validation_sample_50.csv"
df = pd.read_csv(path).fillna("")

# Update the four named email subjects.
updates = [
    ("Defence Drone Technology", "general_announcements"),
    ("Japan Day: Thapar-Japan Roundtable", "general_announcements"),
    ("GRE Masterclass by ETS", "general_announcements"),
    ("International Workshop on Fuzzy Set-Based Modeling", "general_announcements"),
]

changed = 0

for subject_text, category in updates:
    mask = df["subject"].astype(str).str.contains(
        subject_text, case=False, na=False, regex=False
    )
    df.loc[mask, "verified_category"] = category
    changed += int(mask.sum())

# Update the blank-subject student-life notice.
body_mask = df["body"].astype(str).str.contains(
    "proper channels", case=False, na=False, regex=False
)
blank_subject = df["subject"].astype(str).str.strip().eq("")
mask = blank_subject & body_mask

df.loc[mask, "verified_category"] = "hostel_notices"
changed += int(mask.sum())

df.to_csv(path, index=False, encoding="utf-8-sig")
print(f"Updated {changed} rows in {path}")
