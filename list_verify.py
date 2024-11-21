from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
from datetime import datetime, timezone

# Define the scope
SCOPES = ['https://www.googleapis.com/auth/calendar']

# Function to list upcoming events
def list_calendar_events():
    # Load credentials from token.json
    creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    service = build('calendar', 'v3', credentials=creds)

    # Set the time range for upcoming events
    now = datetime.utcnow().isoformat() + 'Z'  # 'Z' indicates UTC time
    print(f"Fetching events starting from: {now}")

    # Fetch the next 10 upcoming events
    events_result = service.events().list(
        calendarId='primary',
        timeMin=now,  # Filter events that start after the current time
        maxResults=10,
        singleEvents=True,
        orderBy='startTime'
    ).execute()
    events = events_result.get('items', [])

    if not events:
        print("No upcoming events found.")
    else:
        print("Upcoming events:")
        for event in events:
            start = event['start'].get('dateTime', event['start'].get('date'))
            print(f"{start}: {event['summary']}")

if __name__ == "__main__":
    list_calendar_events()