
from pathlib import Path

from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

BASE_DIR = Path(__file__).resolve().parent
SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]


def main():
    credentials_file = BASE_DIR / "credentials.json"

    if not credentials_file.exists():
        print("ERROR: credentials.json was not found.")
        print("Place it inside your CampusMail folder.")
        return

    flow = InstalledAppFlow.from_client_secrets_file(
        str(credentials_file),
        SCOPES,
    )

    print("Opening Google sign-in in your browser...")

    credentials = flow.run_local_server(port=0)

    service = build("gmail", "v1", credentials=credentials)

    profile = service.users().getProfile(userId="me").execute()

    print("\nGMAIL CONNECTION SUCCESSFUL")
    print("Authenticated account:", profile.get("emailAddress"))
    print("Total messages in mailbox:", profile.get("messagesTotal"))
    print("\nYour app has read-only Gmail access.")

    # This test does not save the OAuth token to disk.


if __name__ == "__main__":
    main()
