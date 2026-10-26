
from pathlib import Path
import sys

import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# --------------------------------------------------
# PATHS AND CONFIGURATION
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent
INPUT_FILE = ROOT / "outputs" / "ml_labeling_queue.xlsx"
OUTPUT_DIR = ROOT / "outputs"
MODEL_DIR = ROOT / "models"

CATEGORIES = [
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


def make_pipeline(classifier):
    """Create a TF-IDF text classification pipeline."""
    return Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                strip_accents="unicode",
                ngram_range=(1, 2),
                min_df=1,
                max_features=20000,
                sublinear_tf=True,
            ),
        ),
        ("classifier", classifier),
    ])


def main():
    # --------------------------------------------------
    # 1. CHECK INPUT FILE
    # --------------------------------------------------

    if not INPUT_FILE.exists():
        print(f"ERROR: File not found:\n{INPUT_FILE}")
        print("Make sure the Excel file is inside the outputs folder.")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading manually verified email labels...")

    try:
        df = pd.read_excel(INPUT_FILE, sheet_name=0)
    except PermissionError:
        print("ERROR: Close the Excel workbook and run this script again.")
        sys.exit(1)
    except Exception as error:
        print(f"ERROR reading Excel: {error}")
        sys.exit(1)

    df.columns = [str(column).strip().lower() for column in df.columns]

    required_columns = ["subject", "body", "verified_category"]
    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        print("ERROR: Required columns are missing:")
        print(", ".join(missing_columns))
        print("Columns found:", ", ".join(df.columns))
        sys.exit(1)

    # --------------------------------------------------
    # 2. CLEAN AND VALIDATE LABELS
    # --------------------------------------------------

    df["verified_category"] = (
        df["verified_category"]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    # Ignore rows without a verified label.
    df = df[
        df["verified_category"].notna()
        & (df["verified_category"] != "")
    ].copy()

    # Keep only the project's ten valid categories.
    invalid_mask = ~df["verified_category"].isin(CATEGORIES)

    if invalid_mask.any():
        invalid_values = sorted(
            df.loc[invalid_mask, "verified_category"]
            .dropna()
            .unique()
            .tolist()
        )
        print("WARNING: Ignoring rows with unexpected labels:")
        print(invalid_values)

    df = df[df["verified_category"].isin(CATEGORIES)].copy()

    if df.empty:
        print("ERROR: No valid verified labels were found.")
        print("Fill in the verified_category column and save the workbook.")
        sys.exit(1)

    df["subject"] = df["subject"].fillna("").astype(str)
    df["body"] = df["body"].fillna("").astype(str)

    # Combine subject and body into one text field.
    df["text"] = (
        df["subject"].str.strip()
        + " "
        + df["body"].str.strip()
    ).str.strip()

    # Remove rows that contain no email text.
    df = df[df["text"].str.len() > 0].copy()

    if df.empty:
        print("ERROR: No non-empty email subjects or bodies were found.")
        sys.exit(1)

    # --------------------------------------------------
    # 3. SHOW CATEGORY COUNTS
    # --------------------------------------------------

    counts = (
        df["verified_category"]
        .value_counts()
        .reindex(CATEGORIES, fill_value=0)
    )

    print("\nVerified label counts:")
    print(counts.to_string())
    print(f"\nTotal usable labeled emails: {len(df)}")

    if (counts == 0).any():
        missing = counts[counts == 0].index.tolist()
        print("\nERROR: These categories have no labeled emails:")
        print(", ".join(missing))
        print("Add verified examples before training.")
        sys.exit(1)

    # Choose a fold count that works for the smallest category.
    min_category_count = int(counts.min())
    n_splits = min(5, min_category_count)

    if n_splits < 2:
        print("ERROR: Every category needs at least 2 labeled emails.")
        sys.exit(1)

    if (counts != 10).any():
        print(
            "\nWARNING: Some categories do not contain exactly "
            "10 labeled emails."
        )
        print(
            "Check your verified labels to ensure they match the "
            "email content. Training will use the labels as entered."
        )

    print(
        f"\nUsing stratified {n_splits}-fold cross-validation."
    )

    X = df["text"].tolist()
    y = df["verified_category"].tolist()

    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=42,
    )

    # --------------------------------------------------
    # 4. DEFINE MODELS
    # --------------------------------------------------

    models = {
        "Multinomial Naive Bayes": MultinomialNB(alpha=0.5),
        "Logistic Regression": LogisticRegression(
            max_iter=3000,
            random_state=42,
        ),
        "Linear SVM": LinearSVC(random_state=42),
    }

    results = []
    predictions_by_model = {}

    # --------------------------------------------------
    # 5. CROSS-VALIDATED MODEL EVALUATION
    # --------------------------------------------------

    print("\nTraining and evaluating models...\n")

    for model_name, classifier in models.items():
        pipeline = make_pipeline(classifier)

        # Every email is predicted by a model that did not train on it.
        predictions = cross_val_predict(
            pipeline,
            X,
            y,
            cv=cv,
            method="predict",
        )

        predictions_by_model[model_name] = predictions

        metrics = {
            "model": model_name,
            "accuracy": accuracy_score(y, predictions),
            "macro_precision": precision_score(
                y,
                predictions,
                labels=CATEGORIES,
                average="macro",
                zero_division=0,
            ),
            "macro_recall": recall_score(
                y,
                predictions,
                labels=CATEGORIES,
                average="macro",
                zero_division=0,
            ),
            "macro_f1": f1_score(
                y,
                predictions,
                labels=CATEGORIES,
                average="macro",
                zero_division=0,
            ),
        }

        results.append(metrics)

        print(f"Model: {model_name}")
        print(f"  Accuracy:        {metrics['accuracy']:.3f}")
        print(f"  Macro precision: {metrics['macro_precision']:.3f}")
        print(f"  Macro recall:    {metrics['macro_recall']:.3f}")
        print(f"  Macro F1:        {metrics['macro_f1']:.3f}")
        print()

    # --------------------------------------------------
    # 6. SAVE MODEL COMPARISON
    # --------------------------------------------------

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by=["macro_f1", "accuracy"],
        ascending=False,
    ).reset_index(drop=True)

    comparison_path = OUTPUT_DIR / "model_comparison.csv"
    results_df.to_csv(comparison_path, index=False)

    print("=" * 65)
    print("MODEL COMPARISON")
    print("=" * 65)
    print(
        results_df.to_string(
            index=False,
            float_format=lambda value: f"{value:.3f}",
        )
    )

    # Select by macro F1, using accuracy as the tie-breaker.
    best_model_name = results_df.iloc[0]["model"]
    best_predictions = predictions_by_model[best_model_name]

    print(f"\nBest model by cross-validated macro F1: {best_model_name}")

    # --------------------------------------------------
    # 7. SAVE CLASSIFICATION REPORT
    # --------------------------------------------------

    report = classification_report(
        y,
        best_predictions,
        labels=CATEGORIES,
        output_dict=True,
        zero_division=0,
    )

    report_df = pd.DataFrame(report).transpose()

    report_path = OUTPUT_DIR / "best_model_classification_report.csv"
    report_df.to_csv(report_path, index_label="category")

    print("\nBest model classification report:")
    print(
        classification_report(
            y,
            best_predictions,
            labels=CATEGORIES,
            zero_division=0,
        )
    )

    # --------------------------------------------------
    # 8. SAVE CONFUSION MATRIX
    # --------------------------------------------------

    matrix = confusion_matrix(
        y,
        best_predictions,
        labels=CATEGORIES,
    )

    matrix_df = pd.DataFrame(
        matrix,
        index=CATEGORIES,
        columns=CATEGORIES,
    )

    matrix_df.index.name = "actual_category"

    matrix_path = OUTPUT_DIR / "best_model_confusion_matrix.csv"
    matrix_df.to_csv(matrix_path)

    # --------------------------------------------------
    # 9. SAVE CROSS-VALIDATION PREDICTIONS
    # --------------------------------------------------

    prediction_df = pd.DataFrame({
        "subject": df["subject"].tolist(),
        "verified_category": y,
        "cross_validation_prediction": best_predictions,
    })

    prediction_df["correct"] = (
        prediction_df["verified_category"]
        == prediction_df["cross_validation_prediction"]
    )

    predictions_path = OUTPUT_DIR / "best_model_cv_predictions.csv"
    prediction_df.to_csv(predictions_path, index=False)

    # --------------------------------------------------
    # 10. TRAIN WINNING PIPELINE ON ALL VERIFIED EMAILS
    # --------------------------------------------------

    # This final model is for deployment. Evaluation scores above
    # come from cross-validation predictions, not these fitted rows.
    final_pipeline = make_pipeline(models[best_model_name])
    final_pipeline.fit(X, y)

    model_path = MODEL_DIR / "best_email_classifier.joblib"
    joblib.dump(final_pipeline, model_path)

    # Save the category order for reference.
    categories_path = MODEL_DIR / "email_categories.txt"
    categories_path.write_text(
        "\n".join(CATEGORIES) + "\n",
        encoding="utf-8",
    )

    # --------------------------------------------------
    # 11. FINISH
    # --------------------------------------------------

    print("\n" + "=" * 65)
    print("TRAINING COMPLETE")
    print("=" * 65)

    print("\nFiles created:")
    print(f"1. {comparison_path}")
    print(f"2. {report_path}")
    print(f"3. {matrix_path}")
    print(f"4. {predictions_path}")
    print(f"5. {model_path}")
    print(f"6. {categories_path}")

    print(
        "\nNote: The dataset contains only about 100 labeled emails. "
        "Treat the metrics as preliminary and verify labels before "
        "making accuracy claims."
    )


if __name__ == "__main__":
    main()
