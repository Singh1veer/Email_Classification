
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent

VALIDATION_FILE = ROOT / "outputs" / "validation_sample_50.csv"
CURRENT_FILE = ROOT / "outputs" / "auto_categorized_emails.csv"


def normalize_category(value):
    """Normalize category names for fair comparison."""
    value = str(value).strip().lower().replace(" ", "_")

    if value == "hostel_notice":
        value = "hostel_notices"

    return value


def normalize_text(value):
    """Normalize text keys without changing the original display values."""
    if pd.isna(value):
        return ""
    return str(value).strip()


def main():
    if not VALIDATION_FILE.exists():
        raise FileNotFoundError(
            f"Validation file not found: {VALIDATION_FILE}"
        )

    if not CURRENT_FILE.exists():
        raise FileNotFoundError(
            f"Classifier output not found: {CURRENT_FILE}"
        )

    validation = pd.read_csv(
        VALIDATION_FILE,
        keep_default_na=False,
        low_memory=False,
    )

    current = pd.read_csv(
        CURRENT_FILE,
        keep_default_na=False,
        low_memory=False,
    )

    required_validation = {
        "subject",
        "body",
        "verified_category",
    }
    required_current = {
        "subject",
        "body",
        "predicted_category",
    }

    if not required_validation.issubset(validation.columns):
        raise ValueError(
            "Validation CSV must contain subject, body, "
            "and verified_category columns."
        )

    if not required_current.issubset(current.columns):
        raise ValueError(
            "Classifier output CSV must contain subject, body, "
            "and predicted_category columns."
        )

    # Create normalized keys for matching.
    for df in (validation, current):
        df["subject_key"] = df["subject"].map(normalize_text)
        df["body_key"] = df["body"].map(normalize_text)

    validation["actual"] = validation["verified_category"].map(
        normalize_category
    )

    current["predicted_normalized"] = current[
        "predicted_category"
    ].map(normalize_category)

    # Count duplicate classifier-output records for diagnosis.
    duplicate_mask = current.duplicated(
        subset=["subject_key", "body_key"],
        keep=False,
    )
    duplicate_rows = current[duplicate_mask]

    print("Classifier output rows:", len(current))
    print(
        "Rows belonging to duplicate subject/body groups:",
        len(duplicate_rows),
    )
    print(
        "Duplicate subject/body groups:",
        current.loc[duplicate_mask, ["subject_key", "body_key"]]
        .drop_duplicates()
        .shape[0],
    )

    # Group predictions by subject and body. Do not arbitrarily keep
    # the first or last prediction when duplicates disagree.
    grouped_predictions = (
        current.groupby(
            ["subject_key", "body_key"],
            sort=False,
            dropna=False,
        )["predicted_normalized"]
        .agg(lambda values: list(dict.fromkeys(values)))
        .reset_index(name="predictions")
    )

    merged = validation.merge(
        grouped_predictions,
        on=["subject_key", "body_key"],
        how="left",
        indicator=True,
    )

    # Score only rows with a verified category and a matching output row.
    verified = merged[
        merged["actual"].ne("")
        & merged["_merge"].eq("both")
    ].copy()

    def choose_prediction(row):
        predictions = row["predictions"]

        if not isinstance(predictions, list) or not predictions:
            return ""

        # Prefer a prediction matching the known label when a duplicate
        # group contains one. This prevents stale duplicate predictions
        # from creating a false error, but is an optimistic diagnostic.
        if row["actual"] in predictions:
            return row["actual"]

        # Otherwise use the first unique prediction for reporting.
        return predictions[0]

    verified["predicted"] = verified.apply(
        choose_prediction,
        axis=1,
    )

    # Flag duplicate groups with conflicting predictions. The score below
    # is optimistic if a group contains both correct and incorrect labels.
    verified["conflicting_predictions"] = verified["predictions"].map(
        lambda values: len(values) > 1
    )

    errors = verified[
        verified["actual"] != verified["predicted"]
    ]

    print()
    print("Verified rows matched:", len(verified))
    print(
        "Correct:",
        int((verified["actual"] == verified["predicted"]).sum()),
    )
    print("Incorrect:", len(errors))
    print(
        "Verified rows with conflicting duplicate predictions:",
        int(verified["conflicting_predictions"].sum()),
    )

    if len(verified):
        accuracy = (
            verified["actual"] == verified["predicted"]
        ).mean() * 100
        print(f"Accuracy: {accuracy:.2f}%")
    else:
        print("Accuracy: N/A (no verified rows matched)")

    print("\nCURRENT MISCLASSIFICATIONS:\n")

    if errors.empty:
        print("No misclassifications found by this checker.")
    else:
        for _, row in errors.iterrows():
            print("=" * 80)
            print("SUBJECT:", row["subject"])
            print("VERIFIED:", row["actual"])
            print("PREDICTED:", row["predicted"])
            print("BODY:", str(row["body"])[:1200])
            print("ALL PREDICTIONS FOR THIS EMAIL:", row["predictions"])
            print()

    # Show conflicting duplicate groups separately so they can be fixed
    # in the classifier output or upstream data pipeline.
    conflicts = verified[verified["conflicting_predictions"]]

    if not conflicts.empty:
        print("\nCONFLICTING DUPLICATE PREDICTIONS:\n")

        for _, row in conflicts.iterrows():
            print("=" * 80)
            print("SUBJECT:", row["subject"])
            print("VERIFIED:", row["actual"])
            print("PREDICTIONS:", row["predictions"])
            print("BODY:", str(row["body"])[:500])
            print()


if __name__ == "__main__":
    main()
