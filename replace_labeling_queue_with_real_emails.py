"""
CampusMail - Replace the 250 synthetic labeling rows with real emails.

This script is written for the exact folder structure in Email_Classification.zip.

Project structure expected:
Email_Classification/
    data/
        college_emails.csv
    models/
        best_email_classifier.joblib
    outputs/
        ml_labeling_queue_plus_200.xlsx
    replace_labeling_queue_with_real_emails.py

Run from the project root:
    python replace_labeling_queue_with_real_emails.py

What it does:
- Preserves rows 1-100 from ml_labeling_queue_plus_200.xlsx, including
  verified_category and all other existing values.
- Replaces rows 101-350 with 250 unique emails from data/college_emails.csv.
- Uses models/best_email_classifier.joblib to predict each new email.
- Writes the prediction to previous_prediction and suggested_category.
- Leaves verified_category blank and review_status as needs_review.
- Saves a NEW workbook; it does not overwrite your original queue or model.
"""

from pathlib import Path
import re
import unicodedata
import warnings

import joblib
import pandas as pd
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.worksheet.datavalidation import DataValidation

ROOT = Path(__file__).resolve().parent

SOURCE_CSV = ROOT / "data" / "college_emails.csv"
SOURCE_QUEUE = ROOT / "outputs" / "ml_labeling_queue_plus_200.xlsx"
MODEL_PATH = ROOT / "models" / "best_email_classifier.joblib"

OUTPUT_XLSX = ROOT / "outputs" / "ml_labeling_queue_real_emails.xlsx"
OUTPUT_CSV = ROOT / "outputs" / "ml_labeling_queue_real_emails.csv"

SHEET_NAME = "Label Emails"
PRESERVE_FIRST = 100
REPLACE_COUNT = 250
EMAIL_CATEGORIES = [
    "hackathons",
    "exams",
    "hostel_notices",
    "seminars",
    "sports",
    "scholarships",
    "assignments",
    "events",
    "general_announcements",
    "placements",
]


def clean_text(value):
    if pd.isna(value):
        return ""
    return str(value).strip()


def normalize(value):
    value = unicodedata.normalize("NFKC", clean_text(value)).lower()
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def make_email_key(subject, body):
    return normalize(subject) + " || " + normalize(body)


def find_email_columns(df):
    lookup = {str(c).strip().lower(): c for c in df.columns}

    subject_col = next(
        (lookup[name] for name in ("subject", "email_subject", "title")
         if name in lookup),
        None,
    )
    body_col = next(
        (lookup[name] for name in ("body", "email_body", "message", "content", "text")
         if name in lookup),
        None,
    )

    if subject_col is None or body_col is None:
        raise ValueError(
            "The source CSV must contain subject and body columns. "
            f"Found: {list(df.columns)}"
        )
    return subject_col, body_col


def main():
    # Check all paths before doing any work.
    for path in (SOURCE_CSV, SOURCE_QUEUE, MODEL_PATH):
        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found:\n{path}\n"
                "Check the project folder structure in the script header."
            )

    print("1/6 Loading original labeling workbook...")
    queue = pd.read_excel(SOURCE_QUEUE, sheet_name=SHEET_NAME).fillna("")
    if len(queue) < PRESERVE_FIRST:
        raise ValueError(
            f"Expected at least {PRESERVE_FIRST} rows in the queue; "
            f"found {len(queue)}."
        )

    required_queue_columns = {
        "email_id", "subject", "body", "previous_prediction",
        "suggested_category", "verified_category", "review_status",
        "review_notes",
    }
    missing = required_queue_columns.difference(queue.columns)
    if missing:
        raise ValueError(f"Queue is missing columns: {sorted(missing)}")

    # Keep first 100 records and every existing field exactly as supplied.
    first_100 = queue.iloc[:PRESERVE_FIRST].copy()

    print("2/6 Loading real college emails...")
    source = pd.read_csv(SOURCE_CSV, low_memory=False).fillna("")
    subject_col, body_col = find_email_columns(source)
    source["_subject"] = source[subject_col].map(clean_text)
    source["_body"] = source[body_col].map(clean_text)
    source = source[
        (source["_subject"] != "") | (source["_body"] != "")
    ].copy()

    source["_dedupe_key"] = source.apply(
        lambda row: make_email_key(row["_subject"], row["_body"]),
        axis=1,
    )
    source = source.drop_duplicates("_dedupe_key").copy()

    # Exclude the first 100 preserved examples from the candidate pool.
    preserved_keys = {
        make_email_key(row["subject"], row["body"])
        for _, row in first_100.iterrows()
    }
    source = source[
        ~source["_dedupe_key"].isin(preserved_keys)
    ].copy()

    if len(source) < REPLACE_COUNT:
        raise ValueError(
            f"Only {len(source)} unique unused emails are available; "
            f"{REPLACE_COUNT} are required."
        )

    print("3/6 Loading your existing trained classifier...")
    # Suppress only sklearn's version warning; model is still loaded from
    # the project's own artifact. Use the same sklearn version as training
    # if predictions behave unexpectedly.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            message="Trying to unpickle estimator",
        )
        model = joblib.load(MODEL_PATH)

    model_text = (
        source["_subject"].astype(str) + "\n" + source["_body"].astype(str)
    ).tolist()

    print("4/6 Predicting categories for real emails...")
    source["_predicted_category"] = model.predict(model_text).astype(str)

    # Balanced selection: take up to 25 per class, then fill any shortage
    # from remaining real emails. This guarantees 250 when enough data exists.
    selected_parts = []
    model_classes = list(getattr(model, "classes_", EMAIL_CATEGORIES))

    for category in model_classes:
        candidates = source[
            source["_predicted_category"] == str(category)
        ].sort_values("_dedupe_key")
        if not candidates.empty:
            selected_parts.append(candidates.head(25))

    if selected_parts:
        selected = pd.concat(selected_parts, ignore_index=True)
    else:
        selected = source.iloc[0:0].copy()

    selected = selected.drop_duplicates("_dedupe_key")
    if len(selected) < REPLACE_COUNT:
        selected_keys = set(selected["_dedupe_key"])
        remaining = source[
            ~source["_dedupe_key"].isin(selected_keys)
        ].sort_values("_dedupe_key")
        selected = pd.concat(
            [selected, remaining.head(REPLACE_COUNT - len(selected))],
            ignore_index=True,
        )

    selected = selected.head(REPLACE_COUNT).copy()
    if len(selected) != REPLACE_COUNT:
        raise ValueError(
            f"Could select only {len(selected)} real emails, not "
            f"{REPLACE_COUNT}. No output workbook was written."
        )

    print("5/6 Creating the 250 replacement rows...")
    new_rows = []
    for number, (_, email) in enumerate(
        selected.iterrows(), start=PRESERVE_FIRST + 1
    ):
        predicted = email["_predicted_category"]
        new_rows.append({
            "email_id": f"EMAIL_{number:04d}",
            "subject": email["_subject"],
            "body": email["_body"],
            "previous_prediction": predicted,
            "suggested_category": predicted,
            "verified_category": "",  # Must be manually verified.
            "review_status": "needs_review",
            "review_notes": (
                "REAL email from data/college_emails.csv. "
                "Category auto-predicted by best_email_classifier.joblib. "
                "Please verify before using for model training."
            ),
        })

    new_df = pd.DataFrame(new_rows, columns=queue.columns)
    final_df = pd.concat([first_100, new_df], ignore_index=True)

    # Preserve the Instructions sheet if it exists.
    print("6/6 Saving the new Excel and CSV files...")
    with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
        final_df.to_excel(writer, sheet_name=SHEET_NAME, index=False)

        try:
            instructions = pd.read_excel(
                SOURCE_QUEUE, sheet_name="Instructions", header=None
            )
            instructions.to_excel(
                writer, sheet_name="Instructions",
                index=False, header=False
            )
        except ValueError:
            pass

        ws = writer.sheets[SHEET_NAME]
        ws.freeze_panes = "A2"
        ws.auto_filter.ref = ws.dimensions

        # Header styling and readable column widths.
        for cell in ws[1]:
            cell.font = Font(bold=True, color="FFFFFF")
            cell.fill = PatternFill("solid", fgColor="17365D")
            cell.alignment = Alignment(wrap_text=True, vertical="center")

        widths = {
            "A": 16, "B": 48, "C": 75, "D": 24,
            "E": 24, "F": 22, "G": 20, "H": 75,
        }
        for col, width in widths.items():
            ws.column_dimensions[col].width = width

        # Add a dropdown for manual verified labels.
        verified_col = list(final_df.columns).index("verified_category") + 1
        verified_letter = ws.cell(row=1, column=verified_col).column_letter
        validation = DataValidation(
            type="list",
            formula1='"' + ",".join(EMAIL_CATEGORIES) + '"',
            allow_blank=True,
        )
        validation.error = "Choose a category from the dropdown."
        validation.errorTitle = "Invalid category"
        validation.prompt = "Select the correct verified category."
        validation.promptTitle = "Verify this email"
        ws.add_data_validation(validation)
        validation.add(
            f"{verified_letter}{PRESERVE_FIRST + 1}:"
            f"{verified_letter}{PRESERVE_FIRST + REPLACE_COUNT}"
        )

    final_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    # Final integrity checks.
    assert len(final_df) == PRESERVE_FIRST + REPLACE_COUNT
    assert first_100.equals(final_df.iloc[:PRESERVE_FIRST])
    assert final_df.iloc[PRESERVE_FIRST:]["verified_category"].astype(str).str.strip().eq("").all()

    print("\nSUCCESS")
    print(f"Excel created: {OUTPUT_XLSX}")
    print(f"CSV created:   {OUTPUT_CSV}")
    print(f"Original first {PRESERVE_FIRST} rows preserved.")
    print(f"New real emails added: {REPLACE_COUNT}")
    print(f"Total rows: {len(final_df)}")
    print("\nNew 250 emails by predicted category:")
    print(new_df["suggested_category"].value_counts().to_string())
    print("\nAll new verified_category cells are blank for manual review.")


if __name__ == "__main__":
    main()
