from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "outputs" / "auto_categorized_emails.csv"
REVIEW = ROOT / "outputs" / "emails_needing_review.csv"

def choose_column(df, candidates):
    lookup = {str(c).lower(): c for c in df.columns}
    for name in candidates:
        if name.lower() in lookup:
            return lookup[name.lower()]
    return None

def main():
    if not OUTPUT.exists():
        print("Categorized output not found. Run the categorizer first.")
        return

    df = pd.read_csv(OUTPUT).fillna("")
    category = choose_column(df, ["predicted_category"])
    subject = choose_column(df, ["subject", "email_subject"])
    body = choose_column(df, ["body", "email_body", "message", "text"])
    review = choose_column(df, ["review_required"])

    if not category:
        print("Could not find predicted_category in the output CSV.")
        print("Available columns:", list(df.columns))
        return

    print("\n" + "=" * 55)
    print("          CAMPUSMAIL - EMAIL CATEGORIZATION")
    print("=" * 55)
    print(f"Total emails loaded: {len(df)}")
    print("\nCATEGORY SUMMARY")
    print(df[category].value_counts().to_string())

    if review:
        flagged = df[review].astype(str).str.lower().isin(
            ["true", "1", "yes"]
        )
        print(f"\nEmails flagged for review: {int(flagged.sum())}")

    while True:
        print("\nMENU")
        print("1. View sample emails")
        print("2. Search emails by keyword")
        print("3. View emails by category")
        print("4. Open review queue summary")
        print("5. Exit")

        choice = input("Choose an option (1-5): ").strip()

        if choice == "1":
            n = min(5, len(df))
            cols = [c for c in [subject, category] if c]
            if body:
                cols.append(body)
            print(df[cols].head(n).to_string(index=False, max_colwidth=180))

        elif choice == "2":
            keyword = input("Enter a keyword: ").strip()
            if not keyword:
                print("Please enter a keyword.")
                continue
            searchable = df.astype(str).apply(
                lambda col: col.str.contains(
                    keyword, case=False, regex=False, na=False
                )
            ).any(axis=1)
            matches = df[searchable]
            print(f"Matches: {len(matches)}")
            cols = [c for c in [subject, category] if c]
            if body:
                cols.append(body)
            if len(matches):
                print(matches[cols].head(10).to_string(
                    index=False, max_colwidth=180
                ))

        elif choice == "3":
            names = sorted(df[category].astype(str).unique())
            print("Available categories:", ", ".join(names))
            selected = input("Enter category exactly as shown: ").strip()
            matches = df[df[category].astype(str).str.lower() == selected.lower()]
            print(f"Emails in category: {len(matches)}")
            cols = [c for c in [subject, category] if c]
            if body:
                cols.append(body)
            if len(matches):
                print(matches[cols].head(10).to_string(
                    index=False, max_colwidth=180
                ))

        elif choice == "4":
            if not REVIEW.exists():
                print("Review queue file not found.")
                continue
            queue = pd.read_csv(REVIEW).fillna("")
            print(f"Review queue records: {len(queue)}")
            print("Columns:", list(queue.columns))
            print(queue.head(5).to_string(index=False, max_colwidth=140))

        elif choice == "5":
            print("CampusMail demo closed.")
            break

        else:
            print("Invalid choice. Enter a number from 1 to 5.")

if __name__ == "__main__":
    main()
