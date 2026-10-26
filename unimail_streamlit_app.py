from pathlib import Path
import base64
from email.utils import parsedate_to_datetime

import joblib
import pandas as pd
import streamlit as st
from bs4 import BeautifulSoup
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "best_email_classifier.joblib"
CLIENT_SECRET_PATH = BASE_DIR / "credentials.json"
TOKEN_PATH = BASE_DIR / "token.json"
SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]
CATEGORIES = ["hackathons", "exams", "hostel_notices", "seminars", "sports",
              "scholarships", "assignments", "events", "general_announcements", "placements"]

st.set_page_config(page_title="UniMail Classifier", page_icon="📬", layout="wide")


def get_header(headers, wanted):
    for item in headers:
        if item.get("name", "").lower() == wanted.lower():
            return item.get("value", "")
    return ""


def decode_part(data):
    if not data:
        return ""
    data += "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode(data).decode("utf-8", errors="replace")


def extract_body(payload):
    plain, html = [], []

    def walk(part):
        mime = part.get("mimeType", "")
        data = part.get("body", {}).get("data")
        if data and mime == "text/plain":
            plain.append(decode_part(data))
        elif data and mime == "text/html":
            html.append(BeautifulSoup(decode_part(data), "html.parser").get_text(" ", strip=True))
        for child in part.get("parts", []):
            walk(child)

    walk(payload)
    return "\n".join(plain or html)


def get_gmail_service():
    if not CLIENT_SECRET_PATH.exists():
        raise FileNotFoundError("credentials.json is missing from the app folder.")
    creds = None
    # Local single-user demo only. Keep token.json private.
    if TOKEN_PATH.exists():
        creds = Credentials.from_authorized_user_file(str(TOKEN_PATH), SCOPES)
    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
    if not creds or not creds.valid:
        flow = InstalledAppFlow.from_client_secrets_file(str(CLIENT_SECRET_PATH), SCOPES)
        creds = flow.run_local_server(port=0)
        TOKEN_PATH.write_text(creds.to_json(), encoding="utf-8")
    return build("gmail", "v1", credentials=creds)


def fetch_messages(service, max_emails):
    rows, page_token = [], None
    while len(rows) < max_emails:
        batch_size = min(100, max_emails - len(rows))
        result = service.users().messages().list(
            userId="me", maxResults=batch_size, pageToken=page_token,
            includeSpamTrash=False,
        ).execute()
        for item in result.get("messages", []):
            message = service.users().messages().get(
                userId="me", id=item["id"], format="full"
            ).execute()
            payload = message.get("payload", {})
            headers = payload.get("headers", [])
            date = get_header(headers, "Date")
            try:
                date = parsedate_to_datetime(date).strftime("%Y-%m-%d %H:%M")
            except (ValueError, TypeError, OverflowError):
                pass
            rows.append({
                "email_id": message.get("id", ""),
                "date": date,
                "sender": get_header(headers, "From"),
                "subject": get_header(headers, "Subject"),
                "body": extract_body(payload),
            })
        page_token = result.get("nextPageToken")
        if not page_token:
            break
    return rows


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Saved model not found: {MODEL_PATH}")
    # Assumes the joblib file is a complete sklearn Pipeline (TF-IDF + classifier).
    return joblib.load(MODEL_PATH)


def classify_rows(rows, model):
    texts = [(f"{r['subject']} {r['body']}").strip() for r in rows]
    if not texts:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["predicted_category"] = [str(x) for x in model.predict(texts)]
    return df


st.title("📬 UniMail Classifier")
st.write("Connect Gmail, classify messages, and browse your predicted categories.")
st.info("Entering an email address alone cannot grant mailbox access. You must sign in with Google and approve read-only permission. This starter app is for local, single-user testing.")

with st.sidebar:
    st.header("Connect Gmail")
    gmail_address = st.text_input("Gmail address (for reference)", placeholder="you@gmail.com")
    max_emails = st.selectbox("Recent emails to process", [25, 50, 100, 250, 500], index=2)
    connect_clicked = st.button("Sign in with Google & classify", type="primary", use_container_width=True)
    st.caption("The Google account you authorize is the account that will be read.")

if connect_clicked:
    try:
        with st.spinner("Opening Google sign-in..."):
            service = get_gmail_service()
        profile = service.users().getProfile(userId="me").execute()
        actual_email = profile.get("emailAddress", "")
        if gmail_address and gmail_address.strip().lower() != actual_email.lower():
            st.warning(f"You authorized {actual_email}, which differs from the address entered. The authorized account will be used.")
        with st.spinner(f"Fetching up to {max_emails} emails..."):
            emails = fetch_messages(service, max_emails)
        if not emails:
            st.warning("No messages were returned.")
        else:
            with st.spinner("Loading model and classifying emails..."):
                model = load_model()
                results = classify_rows(emails, model)
            st.session_state["unimail_results"] = results
            st.session_state["unimail_account"] = actual_email
            st.success(f"Classified {len(results)} messages for {actual_email}.")
    except Exception as exc:
        st.error(f"Could not connect or classify emails: {exc}")
        st.caption("Check credentials.json, Gmail API enablement, OAuth test-user settings, and whether the saved model is a complete TF-IDF + classifier pipeline.")

df = st.session_state.get("unimail_results")
if df is not None and not df.empty:
    st.subheader(f"Categorized inbox — {st.session_state.get('unimail_account', '')}")
    c1, c2 = st.columns(2)
    c1.metric("Emails processed", len(df))
    c2.metric("Categories represented", df["predicted_category"].nunique())
    left, right = st.columns([1, 2])
    with left:
        st.markdown("#### Emails per category")
        counts = df["predicted_category"].value_counts().reindex(CATEGORIES, fill_value=0)
        st.dataframe(counts.rename_axis("Category").reset_index(name="Emails"), use_container_width=True, hide_index=True)
        st.bar_chart(counts)
    with right:
        st.markdown("#### Browse emails")
        available = sorted(df["predicted_category"].dropna().unique().tolist())
        selected = st.selectbox("Choose category", ["All categories"] + available)
        shown = df if selected == "All categories" else df[df["predicted_category"] == selected]
        st.dataframe(shown[["date", "sender", "subject", "predicted_category"]], use_container_width=True, hide_index=True)
    st.download_button("Download results (CSV)", data=df.to_csv(index=False).encode("utf-8-sig"), file_name="unimail_classified_emails.csv", mime="text/csv")
    st.caption("Predictions may be incorrect. This app displays categories only; it does not create Gmail labels or modify messages.")
