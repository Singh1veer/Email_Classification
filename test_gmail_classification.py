
from pathlib import Path
import base64
from email.utils import parsedate_to_datetime

import joblib
import pandas as pd
from bs4 import BeautifulSoup
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

BASE_DIR = Path(__file__).resolve().parent
SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]
MODEL_PATH = BASE_DIR / "models" / "best_email_classifier.joblib"
TOKEN_PATH = BASE_DIR / "token.json"


def connect_gmail():
    creds = None

    if TOKEN_PATH.exists():
        creds = Credentials.from_authorized_user_file(
            str(TOKEN_PATH), SCOPES
        )

    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())

    if not creds or not creds.valid:
        flow = InstalledAppFlow.from_client_secrets_file(
            str(BASE_DIR / "credentials.json"), SCOPES
        )
        creds = flow.run_local_server(port=0)
        TOKEN_PATH.write_text(creds.to_json(), encoding="utf-8")

    return build("gmail", "v1", credentials=creds)


def decode_part(data):
    if not data:
        return ""
    data += "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode(data).decode(
        "utf-8", errors="replace"
    )


def extract_body(payload):
    """Prefer plain-text parts, with HTML as a fallback."""
    plain_parts = []
    html_parts = []

    def walk(part):
        mime = part.get("mimeType", "")
        data = part.get("body", {}).get("data")

        if data and mime == "text/plain":
            plain_parts.append(decode_part(data))
        elif data and mime == "text/html":
            html = decode_part(data)
            html_parts.append(
                BeautifulSoup(html, "html.parser").get_text(" ", strip=True)
            )

        for child in part.get("parts", []):
            walk(child)

    walk(payload)
    return "\n".join(plain_parts or html_parts)


def header_value(headers, name):
    for header in headers:
        if header.get("name", "").lower() == name.lower():
            return header.get("value", "")
    return ""


def fetch_messages(service, limit=25):
    response = service.users().messages().list(
        userId="me",
        maxResults=limit,
        includeSpamTrash=False,
    ).execute()

    rows = []

    for item in response.get("messages", []):
        msg = service.users().messages().get(
            userId="me",
            id=item["id"],
            format="full",
        ).execute()

        payload = msg.get("payload", {})
        headers = payload.get("headers", [])
        date_value = header_value(headers, "Date")

        try:
            date_value = parsedate_to_datetime(date_value).strftime(
                "%Y-%m-%d %H:%M"
            )
        except (ValueError, TypeError, OverflowError):
            pass

        rows.append({
            "email_id": msg["id"],
            "date": date_value,
            "sender": header_value(headers, "From"),
            "subject": header_value(headers, "Subject"),
            "body": extract_body(payload),
        })

    return rows


def main():
    if not (BASE_DIR / "credentials.json").exists():
        raise FileNotFoundError("credentials.json is missing.")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    print("Loading saved classifier...")
    model = joblib.load(MODEL_PATH)

    print("Connecting to Gmail...")
    service = connect_gmail()

    print("Fetching up to 25 emails...")
    emails = fetch_messages(service, limit=25)

    if not emails:
        print("No messages found.")
        return

    texts = [
        f"{email['subject']} {email['body']}".strip()
        for email in emails
    ]

    print("Classifying emails...")
    predictions = model.predict(texts)

    for email, prediction in zip(emails, predictions):
        email["predicted_category"] = str(prediction)

    df = pd.DataFrame(emails)

    output_path = BASE_DIR / "outputs" / "gmail_classification_test.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print("\nCategory counts:")
    print(df["predicted_category"].value_counts().to_string())

    print("\nSample predictions:")
    print(
        df[["subject", "predicted_category"]]
        .head(10)
        .to_string(index=False)
    )

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
