from pathlib import Path
import sys
import re

import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from xgboost import XGBClassifier


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

    if not INPUT_FILE.exists():
        print("ERROR: Training Excel file not found:", INPUT_FILE)
        sys.exit(1)

    OUTPUT_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)

    print("Loading the 550-email workbook...")

    df = pd.read_excel(INPUT_FILE, sheet_name=0)
    df.columns = [str(c).strip().lower() for c in df.columns]

    required = ["subject", "body", "verified_category"]

    for column in required:
        if column not in df.columns:
            raise ValueError(f"Missing required column: {column}")

    print("Total rows in Excel:", len(df))

    # Keep only manually verified labels.
    df["verified_category"] = (
        df["verified_category"]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    df = df[
        df["verified_category"].notna()
        & (df["verified_category"] != "")
        & df["verified_category"].isin(CATEGORIES)
    ].copy()

    df["subject"] = df["subject"].fillna("").astype(str)
    df["body"] = df["body"].fillna("").astype(str)

    df["text"] = (
        df["subject"].str.strip()
        + " "
        + df["body"].str.strip()
    ).str.strip()

    df = df[df["text"].str.len() > 0].copy()

    if df.empty:
        raise ValueError("No manually verified emails found.")

    counts = (
        df["verified_category"]
        .value_counts()
        .reindex(CATEGORIES, fill_value=0)
    )

    print("\nVerified category counts:")
    print(counts.to_string())

    print("\nTotal usable verified emails:", len(df))

    if (counts == 0).any():
        raise ValueError(
            "At least one category has no verified examples."
        )

    n_splits = min(5, int(counts.min()))

    if n_splits < 2:
        raise ValueError(
            "Every category needs at least two verified emails."
        )

    print(f"\nUsing {n_splits}-fold stratified cross-validation.")

    X = df["text"].tolist()
    y = df["verified_category"].tolist()

    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=42,
    )

    # Keep the existing models and add two new models.
    models = {
        "Multinomial Naive Bayes": MultinomialNB(alpha=0.5),

        "Logistic Regression": LogisticRegression(
            max_iter=3000,
            random_state=42,
        ),

        "Linear SVM": LinearSVC(
            random_state=42,
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        ),

        "XGBoost": OneVsRestClassifier(
            XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                n_estimators=150,
                max_depth=4,
                learning_rate=0.08,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=1,
                random_state=42,
                n_jobs=1,
                verbosity=0,
            ),
            n_jobs=-1,
        ),
    }

    results = []
    all_reports = []
    all_predictions = {}

    print("\nTraining and evaluating all five models...\n")

    for model_name, classifier in models.items():

        print("Evaluating:", model_name)

        pipeline = make_pipeline(classifier)

        predictions = cross_val_predict(
            pipeline,
            X,
            y,
            cv=cv,
            method="predict",
            n_jobs=1,
        )

        all_predictions[model_name] = predictions

        accuracy = accuracy_score(y, predictions)
        precision = precision_score(
            y, predictions, average="macro", zero_division=0
        )
        recall = recall_score(
            y, predictions, average="macro", zero_division=0
        )
        f1 = f1_score(
            y, predictions, average="macro", zero_division=0
        )

        weighted_precision = precision_score(
            y, predictions, average="weighted", zero_division=0
        )
        weighted_recall = recall_score(
            y, predictions, average="weighted", zero_division=0
        )
        weighted_f1 = f1_score(
            y, predictions, average="weighted", zero_division=0
        )

        results.append({
            "Model": model_name,
            "Accuracy": accuracy,
            "Macro Precision": precision,
            "Macro Recall": recall,
            "Macro F1": f1,
            "Weighted Precision": weighted_precision,
            "Weighted Recall": weighted_recall,
            "Weighted F1": weighted_f1,
        })

        report = classification_report(
            y,
            predictions,
            labels=CATEGORIES,
            output_dict=True,
            zero_division=0,
        )

        for category, values in report.items():
            if isinstance(values, dict):
                all_reports.append({
                    "Model": model_name,
                    "Category": category,
                    **values,
                })

        matrix = confusion_matrix(
            y,
            predictions,
            labels=CATEGORIES,
        )

        matrix_df = pd.DataFrame(
            matrix,
            index=CATEGORIES,
            columns=CATEGORIES,
        )

        safe_name = re.sub(
            r"[^a-z0-9]+", "_", model_name.lower()
        ).strip("_")

        matrix_df.to_csv(
            OUTPUT_DIR / f"confusion_matrix_{safe_name}.csv"
        )

        prediction_df = pd.DataFrame({
            "subject": df["subject"].values,
            "verified_category": y,
            "cross_validation_prediction": predictions,
            "correct": [
                actual == predicted
                for actual, predicted in zip(y, predictions)
            ],
        })

        prediction_df.to_csv(
            OUTPUT_DIR / f"predictions_{safe_name}.csv",
            index=False,
        )

        print(
            f"Accuracy: {accuracy:.4f} | "
            f"Macro F1: {f1:.4f}"
        )

    # Save comparison of all models.
    comparison = pd.DataFrame(results)
    comparison = comparison.sort_values(
        "Macro F1", ascending=False
    )

    comparison.to_csv(
        OUTPUT_DIR / "model_comparison.csv",
        index=False,
    )

    pd.DataFrame(all_reports).to_csv(
        OUTPUT_DIR / "classification_reports_all_models.csv",
        index=False,
    )

    # Choose the model with the highest macro F1.
    best_name = comparison.iloc[0]["Model"]
    best_classifier = models[best_name]

    print("\nBest model:", best_name)

    # Fit the winning model on all verified emails.
    best_pipeline = make_pipeline(best_classifier)
    best_pipeline.fit(X, y)

    joblib.dump(
        best_pipeline,
        MODEL_DIR / "best_email_classifier.joblib",
    )

    with open(MODEL_DIR / "email_categories.txt", "w") as f:
        f.write("\n".join(CATEGORIES))

    # Also save every trained model separately.
    for model_name, classifier in models.items():

        safe_name = re.sub(
            r"[^a-z0-9]+", "_", model_name.lower()
        ).strip("_")

        pipeline = make_pipeline(classifier)
        pipeline.fit(X, y)

        joblib.dump(
            pipeline,
            MODEL_DIR / f"{safe_name}_email_classifier.joblib",
        )

    print("\n========== FINAL MODEL COMPARISON ==========")
    print(comparison.to_string(index=False))

    print("\nAll evaluation files saved in outputs/")
    print("All trained models saved in models/")
    print("Best model saved as best_email_classifier.joblib")


if __name__ == "__main__":
    main()