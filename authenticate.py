from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

SCOPES = ['https://www.googleapis.com/auth/calendar']

# Authenticate and save token
def authenticate_google_calendar():
    flow = InstalledAppFlow.from_client_secrets_file(
        'credentials.json', SCOPES
    )
    creds = flow.run_local_server(port=0)
    with open('token.json', 'w') as token_file:
        token_file.write(creds.to_json())
    print("Authentication successful. Token saved to 'token.json'.")

if __name__ == "__main__":
    authenticate_google_calendar()