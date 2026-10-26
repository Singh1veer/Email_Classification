
from pathlib import Path
import pandas as pd
import re

ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = ROOT / "outputs" / "college_emails_clean.csv"
OUTPUT_FILE = ROOT / "outputs" / "auto_categorized_emails.csv"
REVIEW_FILE = ROOT / "outputs" / "emails_needing_review.csv"

# Categories and their keyword rules.
RULES = {
    "exams": [
        r"\bexam(s|ination(s)?)?\b",
        r"\bmid[- ]?term\b",
        r"\bend[- ]?term\b",
        r"\bdate sheet\b",
        r"\bquestion paper\b",
        r"\badmit card\b",
        r"\bexamination schedule\b",
    ],
    "assignments": [
        r"\bassignment(s)?\b",
        r"\bcoursework\b",
        r"\blab submission\b",
        r"\bsubmit your project\b",
        r"\bproject submission\b",
        r"\bpractical file\b",
    ],
    "events": [
        r"\bworkshop(s)?\b",
        r"\bseminar(s)?\b",
        r"\bwebinar(s)?\b",
        r"\bcompetition(s)?\b",
        r"\bfest(ival)?\b",
        r"\bcultural event(s)?\b",
        r"\bguest lecture\b",
        r"\brecruitment drive\b",
    ],
    "deadlines": [
        r"\bdeadline(s)?\b",
        r"\bdue date\b",
        r"\blast date\b",
        r"\bsubmit by\b",
        r"\bpayment due\b",
        r"\bfee payment\b",
        r"\bclosing date\b",
        r"\bregistration closes\b",
    ],
}

def find_matches(text):
    """Return categories whose keywords occur in the email."""
    matches = []

    for category, patterns in RULES.items():
        if any(re.search(pattern, text, flags=re.IGNORECASE)
               for pattern in patterns):
            matches.append(category)

    return matches


def categorize(row):
    subject = str(row["subject"])
    body = str(row["body"])

    # Subject matches carry more weight than body-only matches.
    subject_matches = find_matches(subject)
    body_matches = find_matches(body)

    scores = {}

    for category in RULES:
        scores[category] = (
            2 * (category in subject_matches)
            + 1 * (category in body_matches)
        )

    ranked = sorted(
        scores.items(),
        key=lambda item: item[1],
        reverse=True
    )

    best_category, best_score = ranked[0]
    second_score = ranked[1][1]

    # No rule matched: don't pretend we know the category.
    if best_score == 0:
        return pd.Series({
            "predicted_category": "general_announcements",
            "matched_categories": "",
            "review_required": True,
            "rule_score": 0,
        })

    matched = [
        category for category, score in scores.items()
        if score > 0
    ]

    # Review ties and emails with multiple category matches.
    needs_review = (
        best_score == second_score
        or len(matched) > 1
    )

    return pd.Series({
        "predicted_category": best_category,
        "matched_categories": "; ".join(matched),
        "review_required": needs_review,
        "rule_score": best_score,
    })


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Could not find input file: {INPUT_FILE}"
        )

    df = pd.read_csv(
        INPUT_FILE,
        keep_default_na=False,
        low_memory=False
    )

    required = {"subject", "body"}
    if not required.issubset(df.columns):
        raise ValueError("Input must contain subject and body columns.")

    predictions = df.apply(categorize, axis=1)
    result = pd.concat([df, predictions], axis=1)

    # Save all predictions separately; original remains untouched.
    result.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    review = result[result["review_required"]]
    review.to_csv(
        REVIEW_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    print("\nAutomatic categorization complete.")
    print("Total emails:", len(result))
    print("\nPredicted category counts:")
    print(result["predicted_category"].value_counts().to_string())
    print("\nEmails requiring review:", len(review))
    print("All predictions:", OUTPUT_FILE)
    print("Review queue:", REVIEW_FILE)


if __name__ == "__main__":
    main()
