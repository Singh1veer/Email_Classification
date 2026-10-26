from pathlib import Path
import base64
import html
from datetime import datetime
from email.utils import parsedate_to_datetime, parseaddr

import altair as alt
import joblib
import pandas as pd
import streamlit as st
from bs4 import BeautifulSoup
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

# ───────────────────────────── paths & constants ─────────────────────────────
BASE = Path(__file__).resolve().parent
MODEL_PATH = BASE / "models" / "best_email_classifier.joblib"
CATEGORIES_PATH = BASE / "models" / "email_categories.txt"
CLIENT_PATH = BASE / "credentials.json"
TOKEN_PATH = BASE / "token.json"
METRICS_PATH = BASE / "outputs" / "model_comparison.csv"
REPORT_PATH = BASE / "outputs" / "best_model_classification_report.csv"
CONFUSION_PATH = BASE / "outputs" / "best_model_confusion_matrix.csv"
SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]
DEFAULT_CATEGORIES = ["hackathons", "exams", "hostel_notices", "seminars", "sports", "scholarships",
                      "assignments", "events", "general_announcements", "placements"]

# colour + dot emoji per category (used by tags, charts and the mail list)
CAT_COLORS = {
    "hackathons": "#a78bfa", "exams": "#f87171", "hostel_notices": "#a8845c", "seminars": "#60a5fa",
    "sports": "#34d399", "scholarships": "#fbbf24", "assignments": "#fb923c", "events": "#f472b6",
    "general_announcements": "#94a3b8", "placements": "#22d3ee",
}
CAT_DOTS = {
    "hackathons": "🟣", "exams": "🔴", "hostel_notices": "🟤", "seminars": "🔵", "sports": "🟢",
    "scholarships": "🟡", "assignments": "🟠", "events": "💗", "general_announcements": "⚪", "placements": "🩵",
}
URGENT = {"exams", "assignments", "scholarships", "placements"}   # deadline-sensitive categories
DESCRIPTIONS = {
    "hackathons": "Buildathons, coding contests and innovation challenges",
    "exams": "Exam schedules, datesheets and examination notices",
    "hostel_notices": "Hostel rules, maintenance and resident updates",
    "seminars": "Guest talks, workshops and academic sessions",
    "sports": "Trials, tournaments and sports activities",
    "scholarships": "Financial aid, grants and scholarship deadlines",
    "assignments": "Coursework, submissions and project deadlines",
    "events": "Campus festivals, functions and student activities",
    "general_announcements": "Broad notices and college-wide updates",
    "placements": "Recruitment drives, internships and placement notices",
}

THEMES = {
    "dark": dict(bg="#0b1220", bg2="#0e1627", panel="#162033", panel2="#1b2940", line="rgba(148,170,210,.16)",
                 ink="#e8eefc", muted="#8fa1c4", accent="#3b82f6", accent_soft="rgba(59,130,246,.16)",
                 input="#101a2c", sel="rgba(59,130,246,.18)", hover="rgba(148,170,210,.08)",
                 h1="linear-gradient(90deg,rgba(16,185,129,.28),rgba(20,184,166,.10))",
                 h2="linear-gradient(90deg,rgba(239,68,68,.28),rgba(244,114,182,.10))",
                 h3="linear-gradient(90deg,rgba(139,92,246,.32),rgba(99,102,241,.10))",
                 h4="linear-gradient(90deg,rgba(59,130,246,.30),rgba(34,211,238,.10))", shadow="0 10px 30px rgba(0,0,0,.28)"),
    "light": dict(bg="#eceef2", bg2="#ffffff", panel="#ffffff", panel2="#f5f6f8", line="#e4e7ec",
                  ink="#111827", muted="#6b7280", accent="#1d4ed8", accent_soft="#e0ebff",
                  input="#f3f4f6", sel="#dbeafe", hover="#f3f4f6",
                  h1="linear-gradient(90deg,#d1fae5,#ecfdf5)", h2="linear-gradient(90deg,#fee2e2,#fef2f2)",
                  h3="linear-gradient(90deg,#ede9fe,#f5f3ff)", h4="linear-gradient(90deg,#dbeafe,#eff6ff)",
                  shadow="0 6px 24px rgba(17,24,39,.07)"),
}

st.set_page_config(page_title="CampusMail | Intelligent Inbox", page_icon="✉️", layout="wide",
                   initial_sidebar_state="expanded")

# ───────────────────────────── session state ─────────────────────────────
st.session_state.setdefault("history", [])
st.session_state.setdefault("gmail_df", pd.DataFrame())
st.session_state.setdefault("gmail_account", "")
st.session_state.setdefault("page", "Dashboard")
st.session_state.setdefault("sel_id", None)


def go_inbox():
    st.session_state.page = "Inbox"


def pick(mail_id):
    st.session_state.sel_id = mail_id


# ───────────────────────────── helpers (backend) ─────────────────────────────
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


def categories():
    if CATEGORIES_PATH.exists():
        vals = [x.strip() for x in CATEGORIES_PATH.read_text(encoding="utf-8").splitlines() if x.strip()]
        if vals:
            return vals
    return DEFAULT_CATEGORIES


CATS = categories()


def get_header(headers, wanted):
    for h in headers:
        if h.get("name", "").lower() == wanted.lower():
            return h.get("value", "")
    return ""


def decode_part(data):
    if not data:
        return ""
    data += "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode(data).decode("utf-8", errors="replace")


def extract_body(payload):
    plain, html_parts = [], []

    def walk(part):
        mime = part.get("mimeType", "")
        data = part.get("body", {}).get("data")
        if data and mime == "text/plain":
            plain.append(decode_part(data))
        elif data and mime == "text/html":
            html_parts.append(BeautifulSoup(decode_part(data), "html.parser").get_text(" ", strip=True))
        for child in part.get("parts", []):
            walk(child)

    walk(payload)
    return "\n".join(plain or html_parts)


def gmail_service():
    if not CLIENT_PATH.exists():
        raise FileNotFoundError("credentials.json is missing. Put the downloaded Google OAuth client JSON beside this app.")
    creds = None
    if TOKEN_PATH.exists():
        creds = Credentials.from_authorized_user_file(str(TOKEN_PATH), SCOPES)
    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
    if not creds or not creds.valid:
        flow = InstalledAppFlow.from_client_secrets_file(str(CLIENT_PATH), SCOPES)
        creds = flow.run_local_server(port=0)
        TOKEN_PATH.write_text(creds.to_json(), encoding="utf-8")
    return build("gmail", "v1", credentials=creds)


def fetch_emails(service, limit):
    result_rows, token = [], None
    while len(result_rows) < limit:
        response = service.users().messages().list(
            userId="me", maxResults=min(100, limit - len(result_rows)), pageToken=token, includeSpamTrash=False).execute()
        for item in response.get("messages", []):
            msg = service.users().messages().get(userId="me", id=item["id"], format="full").execute()
            payload, headers = msg.get("payload", {}), msg.get("payload", {}).get("headers", [])
            date = get_header(headers, "Date")
            try:
                date = parsedate_to_datetime(date).strftime("%Y-%m-%d %H:%M")
            except (ValueError, TypeError, OverflowError):
                pass
            result_rows.append({"email_id": msg.get("id", ""), "date": date, "sender": get_header(headers, "From"),
                                "subject": get_header(headers, "Subject"), "body": extract_body(payload)})
        token = response.get("nextPageToken")
        if not token:
            break
    return result_rows


def predict_emails(rows, model):
    if not rows:
        return pd.DataFrame()
    texts = [f"{r.get('subject', '')} {r.get('body', '')}".strip() for r in rows]
    result = pd.DataFrame(rows)
    result["predicted_category"] = [str(x) for x in model.predict(texts)]
    return result


# ───────────────────────────── helpers (UI) ─────────────────────────────
def nice(cat):
    return str(cat).replace("_", " ").title()


def esc(x):
    return html.escape(str(x))


def sender_name(raw):
    name, addr = parseaddr(str(raw))
    return name or addr or "Unknown sender"


def tag(cat):
    c = CAT_COLORS.get(cat, "#94a3b8")
    return f'<span class="tag" style="color:{c};background:{c}1f;border-color:{c}55">{esc(nice(cat))}</span>'


def inject_css(t):
    root = ":root{" + ";".join(f"--{k}:{v}" for k, v in t.items()) + "}"
    css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, .stApp, [data-testid="stSidebar"] * {font-family:'Inter',sans-serif;}
.stApp {background:var(--bg); color:var(--ink);}
[data-testid="stHeader"] {background:transparent;}
.block-container {padding-top:1.4rem; max-width:1500px;}
.stMarkdown, .stMarkdown p, label, [data-testid="stWidgetLabel"] *, [data-testid="stCaptionContainer"], h1, h2, h3, h4 {color:var(--ink);}
[data-testid="stCaptionContainer"] {color:var(--muted);}
h1,h2,h3 {letter-spacing:-.02em; font-weight:700;}

/* sidebar = folder list */
[data-testid="stSidebar"] {background:var(--bg2); border-right:1px solid var(--line);}
[data-testid="stSidebar"] [role="radiogroup"] {gap:2px;}
[data-testid="stSidebar"] [role="radiogroup"] label {padding:10px 14px; border-radius:12px; width:100%; cursor:pointer; margin:0;}
[data-testid="stSidebar"] [role="radiogroup"] label > div:first-child {display:none;}
[data-testid="stSidebar"] [role="radiogroup"] label:hover {background:var(--hover);}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {background:var(--panel); box-shadow:var(--shadow);}
[data-testid="stSidebar"] [role="radiogroup"] label p {font-weight:500; font-size:.95rem; color:var(--ink);}
.side-title {color:var(--muted); font-size:.95rem; margin:14px 4px 6px; font-weight:500;}
.brand {font-size:1.25rem; font-weight:700; letter-spacing:-.02em;}
.brand small {display:block; font-weight:400; font-size:.8rem; color:var(--muted); letter-spacing:0;}
.folder {display:flex; justify-content:space-between; align-items:center; padding:7px 12px; border-radius:10px; font-size:.9rem; color:var(--ink);}
.folder:hover {background:var(--hover);}
.folder .n {color:var(--muted); font-size:.85rem;}
.dot {display:inline-block; width:9px; height:9px; border-radius:50%; margin-right:10px;}

/* buttons */
div.stButton > button, div.stDownloadButton > button {border-radius:12px; font-weight:600; min-height:2.6rem; border:1px solid var(--line); background:var(--panel); color:var(--ink);}
div.stButton > button:hover, div.stDownloadButton > button:hover {border-color:var(--accent); color:var(--accent);}
div.stButton > button[kind="primary"], div.stFormSubmitButton > button[kind="primary"], div.stFormSubmitButton > button {background:var(--accent); color:#fff; border:0; box-shadow:0 8px 20px rgba(37,99,235,.28);}
div.stButton > button[kind="primary"]:hover {color:#fff; filter:brightness(1.08);}

/* inputs */
[data-baseweb="input"], [data-baseweb="base-input"], [data-baseweb="textarea"], textarea, [data-baseweb="select"] > div {background:var(--input) !important; color:var(--ink) !important; border-color:var(--line) !important; border-radius:12px !important;}
input, textarea {color:var(--ink) !important;}
[data-baseweb="popover"] ul, [data-baseweb="popover"] li {background:var(--panel) !important; color:var(--ink) !important;}

/* page header (like the dashboard screenshot) */
.page-head {display:flex; align-items:center; gap:14px; padding:6px 2px 14px;}
.page-icon {width:44px; height:44px; border-radius:12px; background:var(--accent_soft); display:flex; align-items:center; justify-content:center; font-size:1.2rem; border:1px solid var(--line);}
.page-title {font-size:1.7rem; font-weight:700; letter-spacing:-.02em; line-height:1.1;}
.page-sub {color:var(--muted); font-size:.85rem;}
.live {display:inline-flex; align-items:center; gap:8px; padding:9px 14px; border-radius:12px; background:var(--panel); border:1px solid var(--line); font-size:.85rem; font-weight:500; white-space:nowrap;}
.live i {width:7px; height:7px; border-radius:50%; background:#22d3ee; display:inline-block;}

/* coloured-header cards */
.card {background:var(--panel); border:1px solid var(--line); border-radius:20px; box-shadow:var(--shadow); overflow:hidden; margin-bottom:16px;}
.card-head {padding:14px 20px; font-weight:600; font-size:.98rem; border-bottom:1px solid var(--line); display:flex; align-items:center; gap:10px;}
.card-body {padding:16px 20px;}
.h1 {background:var(--h1);} .h2 {background:var(--h2);} .h3 {background:var(--h3);} .h4 {background:var(--h4);}
[class*="st-key-card"] {background:var(--panel); border:1px solid var(--line); border-radius:20px; box-shadow:var(--shadow); overflow:hidden; padding:0 !important; gap:0 !important; margin-bottom:16px;}
[class*="st-key-card"] > div {padding:0 4px;}
[class*="st-key-card"] .card-head {margin:0 -4px 10px; border-radius:0;}
[class*="st-key-card"] [data-testid="stVerticalBlock"] {gap:.4rem;}
[class*="st-key-card"] [data-testid="stElementContainer"]:not(:first-child) {padding:0 14px;}

/* kpi tiles */
.kpi {background:var(--panel); border:1px solid var(--line); border-radius:18px; padding:16px 18px; box-shadow:var(--shadow);}
.kpi-label {color:var(--muted); font-size:.82rem; font-weight:500;}
.kpi-value {font-size:1.7rem; font-weight:700; letter-spacing:-.02em; margin-top:4px;}
.kpi-foot {color:var(--muted); font-size:.78rem; margin-top:2px;}

/* tags */
.tag {display:inline-block; padding:2px 10px; border-radius:999px; border:1px solid; font-size:.74rem; font-weight:600; white-space:nowrap;}
.pill {display:inline-block; border:1px solid var(--line); background:var(--accent_soft); color:var(--accent); padding:4px 10px; border-radius:999px; font-size:.76rem; font-weight:600; margin:2px 6px 2px 0;}
.muted {color:var(--muted); font-size:.88rem;}

/* mail list: each st.button becomes a card */
[class*="st-key-mail"] button {width:100%; text-align:left; justify-content:flex-start; background:var(--panel2); border:1px solid var(--line); border-radius:16px; padding:12px 14px; min-height:0; box-shadow:none; height:auto;}
[class*="st-key-mail"] button p {text-align:left; font-size:.84rem; color:var(--muted); line-height:1.45; margin:0;}
[class*="st-key-mail"] button p strong {color:var(--ink); font-size:.95rem;}
[class*="st-key-mail"] button:hover {border-color:var(--accent); color:inherit;}
[class*="st-key-mailsel"] button {background:var(--sel); border-color:var(--accent);}

/* reading pane */
.reader {background:var(--panel); border:1px solid var(--line); border-radius:20px; padding:24px 26px; box-shadow:var(--shadow);}
.reader h2 {margin:0 0 14px; font-size:1.5rem;}
.sender-row {display:flex; align-items:center; gap:14px; margin-bottom:14px;}
.avatar {width:46px; height:46px; border-radius:50%; background:var(--accent_soft); color:var(--accent); display:flex; align-items:center; justify-content:center; font-weight:700; font-size:1.05rem;}
.sender-name {font-weight:600;} .sender-mail {color:var(--muted); font-size:.85rem;}
.reader-body {white-space:pre-wrap; word-break:break-word; line-height:1.65; font-size:.95rem; max-height:480px; overflow-y:auto; border-top:1px solid var(--line); padding-top:16px;}
[data-testid="stDataFrame"] {border:1px solid var(--line); border-radius:14px; overflow:hidden;}
hr {border-color:var(--line);}
"""
    st.markdown(f"<style>{root}{css.replace('var(--accent_soft)', 'var(--accent_soft)')}</style>", unsafe_allow_html=True)


def page_header(icon, title, sub, live=False):
    left, right = st.columns([6, 2])
    with left:
        st.markdown(f"<div class='page-head'><div class='page-icon'>{icon}</div><div><div class='page-title'>{esc(title)}</div>"
                    f"<div class='page-sub'>{esc(sub)}</div></div></div>", unsafe_allow_html=True)
    if live:
        with right:
            b1, b2 = st.columns([4, 1])
            b1.markdown(f"<div style='text-align:right;padding-top:8px'><span class='live'><i></i>Live • {datetime.now():%H:%M:%S}</span></div>",
                        unsafe_allow_html=True)
            if b2.button("⟳", key="refresh", help="Refresh"):
                st.rerun()


def kpi(label, value, foot=""):
    st.markdown(f"<div class='kpi'><div class='kpi-label'>{esc(label)}</div><div class='kpi-value'>{esc(value)}</div>"
                f"<div class='kpi-foot'>{esc(foot)}</div></div>", unsafe_allow_html=True)


def card_head(icon, title, style="h1"):
    st.markdown(f"<div class='card-head {style}'><span>{icon}</span>{esc(title)}</div>", unsafe_allow_html=True)


def style_chart(ch, t):
    return (ch.configure(background="transparent")
            .configure_axis(labelColor=t["muted"], titleColor=t["muted"], gridColor=t["line"], domainColor=t["line"],
                            gridDash=[3, 3], tickColor=t["line"])
            .configure_view(strokeWidth=0)
            .configure_legend(labelColor=t["muted"], titleColor=t["muted"]))


def cat_scale():
    return alt.Scale(domain=[nice(c) for c in CAT_COLORS], range=list(CAT_COLORS.values()))


# ───────────────────────────── sidebar ─────────────────────────────
with st.sidebar:
    st.markdown("<div class='brand'>✉️ CampusMail<small>Intelligent college inbox</small></div>", unsafe_allow_html=True)
    st.write("")
    st.button("＋  Sync Gmail", type="primary", use_container_width=True, on_click=go_inbox)
    st.markdown("<div class='side-title'>Workspace</div>", unsafe_allow_html=True)
    st.radio("Workspace", ["Dashboard", "Inbox", "Try a Prediction", "Model Lab", "Prediction History", "About the Project"],
             key="page", label_visibility="collapsed")
    gdf = st.session_state.gmail_df
    if not gdf.empty:
        st.markdown("<div class='side-title'>Folders</div>", unsafe_allow_html=True)
        counts_side = gdf.predicted_category.value_counts()
        rows_html = f"<div class='folder'><span>📥 All mail</span><span class='n'>{len(gdf)}</span></div>"
        for c in CATS:
            n = int(counts_side.get(c, 0))
            rows_html += (f"<div class='folder'><span><span class='dot' style='background:{CAT_COLORS.get(c, '#94a3b8')}'></span>"
                          f"{esc(nice(c))}</span><span class='n'>{n}</span></div>")
        st.markdown(rows_html, unsafe_allow_html=True)
    st.divider()
    light = st.toggle("Light theme", key="light")
    st.caption("TF-IDF · Linear SVM · 10 classes\n\nLocal academic prototype — predictions may be imperfect.")

T = THEMES["light" if st.session_state.get("light") else "dark"]
inject_css(T)
page = st.session_state.page

# ═════════════════════════════ DASHBOARD ═════════════════════════════
if page == "Dashboard":
    page_header("✉️", "Dashboard", "email classification dashboard", live=True)
    df = st.session_state.gmail_df

    best_name, best_acc = "Linear SVM", "—"
    if METRICS_PATH.exists():
        try:
            _m = pd.read_csv(METRICS_PATH)
            _b = _m.sort_values("macro_f1", ascending=False).iloc[0]
            best_name, best_acc = str(_b["model"]), f"{_b['accuracy']:.1%}"
        except Exception:
            pass

    k = st.columns(4)
    with k[0]:
        kpi("Emails classified", f"{len(df):,}" if not df.empty else f"{len(st.session_state.history):,}",
            st.session_state.gmail_account or "Current session")
    with k[1]:
        kpi("Categories found", str(df.predicted_category.nunique()) if not df.empty else "0", f"of {len(CATS)} supported")
    with k[2]:
        kpi("Best model", best_name, "Selected by macro F1")
    with k[3]:
        kpi("CV accuracy", best_acc, "Small labeled sample")
    st.write("")

    if df.empty:
        st.markdown("<div class='card'><div class='card-head h4'><span>📭</span>No emails classified yet</div><div class='card-body'>"
                    "<p class='muted'>Sync your Gmail (read-only) to see category charts, priorities and a timeline. "
                    "You can also test a single message on the <b>Try a Prediction</b> page.</p></div></div>", unsafe_allow_html=True)
        st.button("Open Inbox & sync", type="primary", on_click=go_inbox)
        st.markdown("#### Supported categories")
        for start in range(0, len(CATS), 2):
            pair = st.columns(2)
            for j, c in enumerate(CATS[start:start + 2]):
                with pair[j]:
                    st.markdown(f"<div class='card'><div class='card-body'>{tag(c)}"
                                f"<div class='muted' style='margin-top:8px'>{esc(DESCRIPTIONS.get(c, 'College email category'))}</div></div></div>",
                                unsafe_allow_html=True)
    else:
        counts = df.predicted_category.value_counts().reindex(CATS, fill_value=0)
        cdf = pd.DataFrame({"Category": [nice(c) for c in counts.index], "Emails": counts.values})
        urgent_n = int(df.predicted_category.isin(URGENT).sum())
        normal_n = len(df) - urgent_n

        c1, c2 = st.columns(2)
        with c1:
            with st.container(key="card_bar"):
                card_head("📄", "Emails by Category", "h1")
                bar = alt.Chart(cdf).mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6).encode(
                    x=alt.X("Category:N", sort=None, axis=alt.Axis(labelAngle=-40, title=None)),
                    y=alt.Y("Emails:Q", title=None),
                    color=alt.Color("Category:N", scale=cat_scale(), legend=None),
                    tooltip=["Category", "Emails"]).properties(height=330)
                st.altair_chart(style_chart(bar, T), use_container_width=True, theme=None)
        with c2:
            with st.container(key="card_donut"):
                card_head("⚠️", "Emails by Priority", "h2")
                pdf = pd.DataFrame({"Priority": ["High", "Normal"], "Emails": [urgent_n, normal_n]})
                donut = alt.Chart(pdf).mark_arc(innerRadius=70, outerRadius=115, stroke="#fff", strokeWidth=1.5).encode(
                    theta="Emails:Q",
                    color=alt.Color("Priority:N", scale=alt.Scale(domain=["High", "Normal"], range=["#ef4444", "#10b981"]), legend=None),
                    tooltip=["Priority", "Emails"]).properties(height=330)
                st.altair_chart(style_chart(donut, T), use_container_width=True, theme=None)
                st.markdown(f"<div style='text-align:center;padding-bottom:14px;font-weight:600;font-size:.9rem'>"
                            f"<span style='color:#ef4444'>●</span> High ({urgent_n}) &nbsp;&nbsp; "
                            f"<span style='color:#10b981'>●</span> Normal ({normal_n})</div>", unsafe_allow_html=True)
                st.caption("High = exams, assignments, scholarships and placements (deadline-sensitive).")

        with st.container(key="card_timeline"):
            card_head("📈", "Classification Timeline", "h3")
            tl = df.copy()
            tl["when"] = pd.to_datetime(tl["date"], errors="coerce")
            tl = tl.dropna(subset=["when"])
            if tl.empty:
                st.caption("No parsable dates to draw a timeline.")
            else:
                tl["day"] = tl["when"].dt.normalize()
                tl["Category"] = tl["predicted_category"].map(nice)
                tdf = tl.groupby(["day", "Category"]).size().reset_index(name="Emails")
                area = alt.Chart(tdf).mark_area(opacity=.75, line=True, interpolate="monotone").encode(
                    x=alt.X("day:T", title=None), y=alt.Y("Emails:Q", stack=True, title=None),
                    color=alt.Color("Category:N", scale=cat_scale(), legend=alt.Legend(orient="bottom", columns=5)),
                    tooltip=["day:T", "Category", "Emails"]).properties(height=260)
                st.altair_chart(style_chart(area, T), use_container_width=True, theme=None)

# ═════════════════════════════ INBOX ═════════════════════════════
elif page == "Inbox":
    df = st.session_state.gmail_df
    page_header("📥", "Inbox", f"{len(df):,} mails classified" if not df.empty else "Connect Gmail to classify your mail")

    with st.expander("Connect Gmail (read-only)", expanded=df.empty):
        st.caption("Google sign-in is required — entering an address alone grants no access. This OAuth flow is for local "
                   "single-user testing; do not deploy it as a multi-user service as-is.")
        a, b, c = st.columns([2, 2, 1.2])
        entered_email = a.text_input("Gmail address (optional reference)", placeholder="student@gmail.com")
        limit = b.select_slider("Messages to process", options=[25, 50, 100, 250, 500, 1000], value=100)
        c.write("")
        c.write("")
        run = c.button("🔐 Sign in & classify", type="primary", use_container_width=True)
        st.caption("Start with 25–100 messages to validate the model. Each message is fetched individually from Gmail.")

    if run:
        try:
            with st.spinner("Opening Google authorization..."):
                service = gmail_service()
            profile = service.users().getProfile(userId="me").execute()
            account = profile.get("emailAddress", "")
            if entered_email.strip() and entered_email.strip().lower() != account.lower():
                st.warning(f"The authorized account is {account}; that account will be used.")
            with st.spinner(f"Fetching up to {limit} recent messages..."):
                rows = fetch_emails(service, int(limit))
            with st.spinner("Loading model and predicting categories..."):
                model = load_model()
                df = predict_emails(rows, model)
            st.session_state.gmail_df = df
            st.session_state.gmail_account = account
            st.session_state.sel_id = None
            st.success(f"Classified {len(df)} messages from {account}.")
            st.rerun()
        except Exception as e:
            st.error(f"Could not connect or classify: {e}")
            st.caption("Check credentials.json, the Gmail API, OAuth test-user configuration and the saved model pipeline.")

    df = st.session_state.gmail_df
    if df.empty:
        st.markdown("<div class='card'><div class='card-head h4'><span>📭</span>Your inbox is empty</div><div class='card-body'>"
                    "<p class='muted'>Open <b>Connect Gmail</b> above and sign in to classify your recent messages.</p></div></div>",
                    unsafe_allow_html=True)
    else:
        # top bar: search + category filter + export (like the search bar in the references)
        s1, s2, s3 = st.columns([3, 1.4, 1.2])
        search_text = s1.text_input("Search", placeholder="🔍  Search sender or subject…", label_visibility="collapsed", key="inbox_search")
        selected = s2.selectbox("Category", ["All categories"] + [c for c in CATS if c in set(df.predicted_category)],
                                format_func=lambda c: c if c == "All categories" else nice(c), label_visibility="collapsed")
        s3.download_button("⬇ Export CSV", df.to_csv(index=False).encode("utf-8-sig"), "campusmail_categorized_emails.csv",
                           "text/csv", use_container_width=True)

        view = df if selected == "All categories" else df[df.predicted_category == selected]
        if search_text.strip():
            needle = search_text.strip().lower()
            view = view[view.apply(lambda r: needle in str(r.get("sender", "")).lower() or needle in str(r.get("subject", "")).lower(), axis=1)]
        view = view.reset_index(drop=True)

        if view.empty:
            st.info("No emails match these filters.")
        else:
            ids = view["email_id"].tolist()
            if st.session_state.sel_id not in ids:
                st.session_state.sel_id = ids[0]
            chosen = view[view.email_id == st.session_state.sel_id].iloc[0]

            list_col, read_col = st.columns([1, 1.7], gap="medium")
            with list_col:
                st.caption(f"Showing {min(len(view), 80)} of {len(view):,} messages")
                with st.container(height=640, border=False):
                    for i, r in view.head(80).iterrows():
                        cat = r["predicted_category"]
                        label = (f"**{sender_name(r['sender'])[:32]}** · {str(r['date'])[:10]}\n\n"
                                 f"{str(r['subject'] or '(No subject)')[:70]}\n\n"
                                 f"{CAT_DOTS.get(cat, '⚪')} {nice(cat)}")
                        key = ("mailsel_" if r["email_id"] == st.session_state.sel_id else "mail_") + str(i)
                        st.button(label, key=key, on_click=pick, args=(r["email_id"],), use_container_width=True)
            with read_col:
                name = sender_name(chosen["sender"])
                _, addr = parseaddr(str(chosen["sender"]))
                body = str(chosen.get("body", "")) or "(No message body available)"
                st.markdown(
                    f"<div class='reader'><h2>{esc(chosen['subject'] or '(No subject)')}</h2>"
                    f"<div class='sender-row'><div class='avatar'>{esc(name[:1].upper())}</div>"
                    f"<div style='flex:1'><div class='sender-name'>{esc(name)}</div><div class='sender-mail'>{esc(addr)}</div></div>"
                    f"<div style='text-align:right'>{tag(chosen['predicted_category'])}<div class='sender-mail' style='margin-top:6px'>{esc(chosen['date'])}</div></div></div>"
                    f"<div class='reader-body'>{esc(body[:12000])}</div></div>", unsafe_allow_html=True)
                st.caption("The category is a model prediction, not a verified label. CampusMail never modifies Gmail.")

# ═════════════════════════════ TRY A PREDICTION ═════════════════════════════
elif page == "Try a Prediction":
    page_header("✨", "Try a Prediction", "paste any email and see the predicted campus category")
    try:
        model = load_model()
    except Exception as e:
        st.error(str(e))
        st.stop()
    f1, f2 = st.columns([1.3, 1], gap="large")
    with f1:
        with st.form("predict_form"):
            subject = st.text_input("Email subject", placeholder="End-semester examination timetable released")
            body = st.text_area("Email body", placeholder="Paste the message body here…", height=220)
            submitted = st.form_submit_button("✨ Predict category", type="primary", use_container_width=True)
    with f2:
        if submitted:
            text = f"{subject} {body}".strip()
            if not text:
                st.warning("Enter a subject or body first.")
            else:
                prediction = str(model.predict([text])[0])
                st.markdown(f"<div class='card'><div class='card-head h3'><span>🎯</span>Model prediction</div><div class='card-body'>"
                            f"<div style='font-size:1.9rem;font-weight:700;letter-spacing:-.02em;margin-bottom:10px'>{esc(nice(prediction))}</div>"
                            f"{tag(prediction)}<p class='muted' style='margin-top:12px'>{esc(DESCRIPTIONS.get(prediction, ''))}</p></div></div>",
                            unsafe_allow_html=True)
                st.session_state.history.append({"Subject": subject, "Email Body": body, "Predicted Category": prediction})
                if hasattr(model, "decision_function"):
                    st.caption("Decision scores are not calibrated probabilities, so no confidence percentage is shown.")
        else:
            st.markdown("<div class='card'><div class='card-head h3'><span>🎯</span>Model prediction</div><div class='card-body'>"
                        "<p class='muted'>Your result will appear here.</p></div></div>", unsafe_allow_html=True)

# ═════════════════════════════ MODEL LAB ═════════════════════════════
elif page == "Model Lab":
    page_header("🧪", "Model Lab", "compare the tested algorithms and inspect saved evaluation outputs")
    if METRICS_PATH.exists():
        metrics = pd.read_csv(METRICS_PATH)
        st.caption("Cross-validation results from the project outputs. The labeled dataset is small (100 examples), "
                   "so treat these numbers as indicative, not production-grade.")
        if not metrics.empty:
            best = metrics.sort_values("macro_f1", ascending=False).iloc[0]
            k = st.columns(4)
            with k[0]:
                kpi("Top model", str(best["model"]), "Highest macro F1")
            with k[1]:
                kpi("Accuracy", f"{best['accuracy']:.1%}", "CV score")
            with k[2]:
                kpi("Macro F1", f"{best['macro_f1']:.3f}", "Equal weight per class")
            with k[3]:
                kpi("Models tested", str(len(metrics)), "Same experiment output")
            st.write("")
        m1, m2 = st.columns([1, 1])
        with m1:
            with st.container(key="card_cmp_table"):
                card_head("📋", "Model comparison", "h1")
                display = metrics.copy()
                for col in ["accuracy", "macro_precision", "macro_recall", "macro_f1"]:
                    if col in display:
                        display[col] = display[col].map(lambda x: f"{x:.3f} ({x:.1%})")
                st.dataframe(display, use_container_width=True, hide_index=True)
        with m2:
            with st.container(key="card_cmp_chart"):
                card_head("📊", "Score comparison", "h4")
                long = metrics.melt(id_vars="model", value_vars=[c for c in ["accuracy", "macro_precision", "macro_recall", "macro_f1"] if c in metrics],
                                    var_name="Metric", value_name="Score")
                ch = alt.Chart(long).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
                    x=alt.X("model:N", title=None, axis=alt.Axis(labelAngle=0)), y=alt.Y("Score:Q", title=None),
                    xOffset="Metric:N", color=alt.Color("Metric:N", legend=alt.Legend(orient="bottom")),
                    tooltip=["model", "Metric", alt.Tooltip("Score:Q", format=".3f")]).properties(height=280)
                st.altair_chart(style_chart(ch, T), use_container_width=True, theme=None)

        e1, e2 = st.columns(2)
        with e1:
            st.markdown("<div class='card'><div class='card-body'><b>Accuracy</b><p class='muted'>Share of all evaluated messages given the correct category. Can hide poor performance on rarer classes.</p>"
                        "<b>Precision</b><p class='muted'>Of messages predicted as a category, how many were correct.</p></div></div>", unsafe_allow_html=True)
        with e2:
            st.markdown("<div class='card'><div class='card-body'><b>Recall</b><p class='muted'>Of the messages truly in a category, how many the model found.</p>"
                        "<b>Macro F1</b><p class='muted'>Balance of precision and recall, averaged equally across categories.</p></div></div>", unsafe_allow_html=True)
    else:
        st.info("Model comparison file not found: outputs/model_comparison.csv")

    if REPORT_PATH.exists():
        report = pd.read_csv(REPORT_PATH)
        report = report[~report.category.isin(["accuracy", "macro avg", "weighted avg"])]
        with st.container(key="card_report"):
            card_head("🏷️", "Per-category performance — Linear SVM", "h3")
            st.caption("F1 helps spot categories that need more verified training examples.")
            r1, r2 = st.columns([1, 1.2])
            r1.dataframe(report, use_container_width=True, hide_index=True)
            r2.bar_chart(report.set_index("category")[["precision", "recall", "f1-score"]])
    if CONFUSION_PATH.exists():
        with st.expander("Inspect confusion matrix"):
            cm = pd.read_csv(CONFUSION_PATH, index_col=0)
            st.caption("Rows are actual categories; columns are predicted. Off-diagonal counts are errors.")
            st.dataframe(cm, use_container_width=True)

    st.markdown("#### Pipeline")
    steps = [("Email text", "Subject and body are combined into one string."), ("TF-IDF", "Text becomes numeric features."),
             ("Unigrams + bigrams", "Training used ngram_range=(1, 2)."),
             ("Algorithm comparison", "Multinomial Naive Bayes, Logistic Regression and Linear SVM were compared."),
             ("Evaluation", "Accuracy, macro precision, macro recall and macro F1 are reported.")]
    cols = st.columns(len(steps))
    for i, (title, desc) in enumerate(steps):
        with cols[i]:
            st.markdown(f"<div class='card'><div class='card-body'><span class='pill'>{i + 1}</span><b>{esc(title)}</b>"
                        f"<div class='muted' style='margin-top:6px'>{esc(desc)}</div></div></div>", unsafe_allow_html=True)

# ═════════════════════════════ HISTORY ═════════════════════════════
elif page == "Prediction History":
    page_header("🕘", "Prediction History", "emails classified on the Try a Prediction page this session")
    hist = pd.DataFrame(st.session_state.history)
    if hist.empty:
        st.info("No manual predictions yet. Open Try a Prediction to begin.")
    else:
        kpi("Predictions this session", str(len(hist)))
        st.write("")
        st.dataframe(hist, use_container_width=True, hide_index=True)
        h1, h2, _ = st.columns([1, 1, 4])
        h1.download_button("Download CSV", hist.to_csv(index=False).encode("utf-8-sig"), "campusmail_prediction_history.csv", "text/csv")
        if h2.button("Clear history"):
            st.session_state.history = []
            st.rerun()

# ═════════════════════════════ ABOUT ═════════════════════════════
elif page == "About the Project":
    page_header("ℹ️", "About the Project", "a machine-learning classifier for college email")
    a1, a2 = st.columns(2)
    with a1:
        st.markdown("<div class='card'><div class='card-head h1'><span>🎯</span>Objectives</div><div class='card-body'><ul class='muted'>"
                    "<li>Reduce time spent scanning college email updates.</li><li>Group messages into ten useful campus categories.</li>"
                    "<li>Compare classical machine-learning models with evaluation metrics.</li>"
                    "<li>Provide read-only Gmail integration for a practical demonstration.</li></ul></div></div>", unsafe_allow_html=True)
    with a2:
        st.markdown("<div class='card'><div class='card-head h2'><span>⚠️</span>Limitations & responsible use</div><div class='card-body'><ul class='muted'>"
                    "<li>Predictions can be wrong; check important deadlines manually.</li>"
                    "<li>Results come from a small labeled sample and may not generalize.</li>"
                    "<li>Gmail access is OAuth read-only; this demo stores a token file locally.</li>"
                    "<li>Never commit credentials.json or token.json. Multi-user deployment needs a proper web OAuth flow and per-user token storage.</li></ul></div></div>",
                    unsafe_allow_html=True)
    st.markdown("#### Categories")
    st.markdown("".join(tag(c) + " " for c in CATS), unsafe_allow_html=True)

st.markdown("<hr><div class='muted' style='text-align:center;padding:8px'>CampusMail Classifier · Academic prototype · TF-IDF + supervised machine learning</div>",
            unsafe_allow_html=True)