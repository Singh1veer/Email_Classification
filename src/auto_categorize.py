
from pathlib import Path
import re

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = ROOT / "outputs" / "college_emails_clean.csv"
OUTPUT_FILE = ROOT / "outputs" / "auto_categorized_emails.csv"
REVIEW_FILE = ROOT / "outputs" / "emails_needing_review.csv"


RULES = {
    "hostel_notices": [
        r"\bhostel(s)?\b",
        r"\bfretbox\b",
        r"\broom key(s)?\b",
        r"\bac remote(s)?\b",
        r"\bhostel allotment\b",
        r"\bhostel accommodation\b",
        r"\bhostel fee(s)?\b",
        r"\broom booking\b",
        r"\bhostel preference(s)?\b",
        r"\bvacat(e|ing|ion)\b",
        r"\bmess fee(s)?\b",
        r"\bhostel executive committee\b",
        r"\broom allocation\b",
    ],
    "exams": [
        r"\bexam(s|ination(s)?)?\b",
        r"\bmid[- ]?term\b",
        r"\bend[- ]?term\b",
        r"\bdate sheet\b",
        r"\bquestion paper\b",
        r"\badmit card\b",
        r"\bexamination schedule\b",
        r"\bEST(s)?\b",
        r"\bEST exam(s)?\b",
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
        r"\bcompetition(s)?\b",
        r"\bfest(ival)?\b",
        r"\bcultural event(s)?\b",
        r"\bguest lecture\b",
        r"\brecruitment drive\b",
        r"\bACE Talks?\b",
        r"\bclash or cash\b",
        r"\brubaru\b",
        r"\bmudranite\b",
        r"\bconcert(s)?\b",
        r"\bperformances?\b",
        r"\bopen mic\b",
        r"\bguest speaker\b",
        r"\bregister now\b",
        r"\bjoin us\b",
        r"\bcelebration(s)?\b",
        r"\baudition(s)?\b",
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
        r"\bapply by\b",
        r"\bapplication deadline\b",
        r"\binternship opportunit(y|ies)\b",
        r"\bkindly share\b",
    ],
}


def category_score(category, subject, body):
    """Score subject evidence more strongly than body-only evidence."""
    subject_hit = any(
        re.search(pattern, subject, flags=re.IGNORECASE)
        for pattern in RULES[category]
    )
    body_hit = any(
        re.search(pattern, body, flags=re.IGNORECASE)
        for pattern in RULES[category]
    )

    subject_weight = 4 if category == "hostel_notices" else 3
    body_weight = 2 if category == "hostel_notices" else 1

    return subject_weight * subject_hit + body_weight * body_hit


def result_row(category, matched, review, score):
    """Return prediction fields in a consistent format."""
    return pd.Series({
        "predicted_category": category,
        "matched_categories": matched,
        "review_required": review,
        "rule_score": score,
    })


def categorize(row):
    subject = "" if pd.isna(row["subject"]) else str(row["subject"])
    body = "" if pd.isna(row["body"]) else str(row["body"])
    combined = f"{subject}\n{body}"

    # Shortlisted internship-program emails are announcements.
    if (
        re.search(r"\bshortlisted\b", combined, flags=re.IGNORECASE)
        and re.search(
            r"\binternship program\b",
            combined,
            flags=re.IGNORECASE,
        )
    ):
        return result_row(
            "general_announcements",
            "general_announcements",
            False,
            5,
        )

    # Specific international academic workshop classified as an event.
    if (
        re.search(
            r"\binternational workshop\b",
            combined,
            flags=re.IGNORECASE,
        )
        and (
            re.search(r"\bfuzzy set\b", combined, flags=re.IGNORECASE)
            or re.search(
                r"\boptimization\b",
                combined,
                flags=re.IGNORECASE,
            )
        )
    ):
        return result_row("events", "events", False, 5)

    # Routine workshops and training sessions default to announcements.
    training_or_workshop = re.search(
        r"\b(workshops?|training sessions?|training programs?|"
        r"aptitude[- ]building|financial literacy|preparation batch|"
        r"demo sessions?)\b",
        combined,
        flags=re.IGNORECASE,
    )

    clear_standalone_event = re.search(
        r"\b(competition|festival|fest|concert|open mic|audition|"
        r"cultural event|literary evening|clash or cash|rubaru|mudranite|"
        r"celebration)\b",
        combined,
        flags=re.IGNORECASE,
    )

    if training_or_workshop and not clear_standalone_event:
        return result_row(
            "general_announcements",
            "general_announcements",
            False,
            4,
        )

    # Payment notices with clear due-date wording are deadlines.
    if (
        re.search(
            r"\b(payment|pay|fee payment)\b",
            subject,
            flags=re.IGNORECASE,
        )
        and re.search(
            r"\b(mess fee|hostel fee)\b",
            combined,
            flags=re.IGNORECASE,
        )
        and re.search(
            r"\b(deadline|last date|due date|pay(?:ment)? by|"
            r"is \d{1,2}(?:st|nd|rd|th)?\s+\w+)\b",
            combined,
            flags=re.IGNORECASE,
        )
    ):
        return result_row(
            "deadlines",
            "deadlines; hostel_notices",
            True,
            4,
        )

    # Institutional news can contain incidental exam wording.
    if re.search(
        r"\bTIET PATIALA IN THE NEWS\b",
        combined,
        flags=re.IGNORECASE,
    ):
        return result_row(
            "general_announcements",
            "general_announcements",
            False,
            4,
        )

    # Campus-wide DoSA student-life notices take priority over
    # hostel keywords. Allow Markdown asterisks and flexible whitespace.
    dosa_student_life_notice = (
        re.search(
            r"Students\s+are\s+the\s+\*?Life\s+of\s+our\s+Campus",
            combined,
            flags=re.IGNORECASE,
        )
        and re.search(
            r"\bDoSA\s+office\b",
            combined,
            flags=re.IGNORECASE,
        )
    )

    if dosa_student_life_notice:
        return result_row(
            "general_announcements",
            "general_announcements",
            False,
            5,
        )

    # Score the remaining categories.
    scores = {
        category: category_score(category, subject, body)
        for category in RULES
    }

    ranked = sorted(
        scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    best_category, best_score = ranked[0]
    second_score = ranked[1][1]

    matched = [
        category
        for category, score in scores.items()
        if score > 0
    ]

    if best_score == 0:
        return result_row(
            "general_announcements",
            "",
            True,
            0,
        )

    needs_review = (
        len(matched) > 1
        or best_score == second_score
        or best_score == 1
    )

    return result_row(
        best_category,
        "; ".join(matched),
        needs_review,
        best_score,
    )


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Could not find input file: {INPUT_FILE}"
        )

    df = pd.read_csv(
        INPUT_FILE,
        keep_default_na=False,
        low_memory=False,
    )

    required = {"subject", "body"}
    if not required.issubset(df.columns):
        raise ValueError(
            "Input must contain subject and body columns."
        )

    predictions = df.apply(categorize, axis=1)

    prediction_columns = [
        "predicted_category",
        "matched_categories",
        "review_required",
        "rule_score",
    ]

    # Remove old prediction columns before adding updated predictions.
    base = df.drop(
        columns=[
            column
            for column in prediction_columns
            if column in df.columns
        ]
    )

    result = pd.concat([base, predictions], axis=1)

    result.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    review = result[result["review_required"]]
    review.to_csv(
        REVIEW_FILE,
        index=False,
        encoding="utf-8-sig",
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
