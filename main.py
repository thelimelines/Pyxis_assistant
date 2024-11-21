import openai
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
from dotenv import load_dotenv
import os
from datetime import datetime
import json

# Load environment variables from .env file
load_dotenv()

# Fetch OpenAI API key from environment variables
openai.api_key = os.getenv("OPENAI_API_KEY")

# Define the scope for Google Calendar API
SCOPES = ['https://www.googleapis.com/auth/calendar']

# Function to interpret event details using GPT
def interpret_event(conversation):
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=conversation
    )
    # Extract the assistant's reply
    message = response.choices[0].message
    return message.content

# Function to add an event to Google Calendar
def add_calendar_event(event_json):
    # Load credentials from token.json
    creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    service = build('calendar', 'v3', credentials=creds)

    # Load event data from JSON
    event_data = json.loads(event_json)

    # Build the event object
    event = {
        'summary': event_data['summary'],
        'start': {
            'dateTime': event_data['start_time'],
            'timeZone': event_data.get('time_zone', 'UTC'),
        },
        'end': {
            'dateTime': event_data['end_time'],
            'timeZone': event_data.get('time_zone', 'UTC'),
        },
    }

    # Add recurrence if provided
    if 'recurrence' in event_data:
        event['recurrence'] = [event_data['recurrence']]

    # Insert the event into the calendar
    created_event = service.events().insert(calendarId='primary', body=event).execute()
    print(f"Event created: {created_event.get('htmlLink')}")

if __name__ == "__main__":
    conversation = [
        {
            "role": "system",
            "content": (
                "You are an assistant that helps interpret calendar events. "
                "Your goal is to extract all necessary details to create a calendar event, "
                "including handling ambiguities by asking the user for clarification. "
                "Please output the final event details in a JSON format with keys: "
                "'summary', 'start_time', 'end_time', 'time_zone' (optional), and 'recurrence' (optional). "
                "Dates and times should be in ISO 8601 format (e.g., '2023-10-13T13:30:00'). "
                "If any information is missing or ambiguous, ask the user specific questions to clarify."
            ),
        },
    ]

    print("Describe the event you'd like to add:")
    user_input = input()
    conversation.append({"role": "user", "content": user_input})

    while True:
        # Use GPT to interpret the event details
        assistant_reply = interpret_event(conversation)
        print(f"Assistant: {assistant_reply}")
        conversation.append({"role": "assistant", "content": assistant_reply})

        # Try to parse the assistant's response as JSON
        try:
            event_json_start = assistant_reply.find('{')
            event_json_end = assistant_reply.rfind('}') + 1
            event_json_str = assistant_reply[event_json_start:event_json_end]
            event_data = json.loads(event_json_str)
            # If parsing is successful, attempt to add the event
            add_calendar_event(event_json_str)
            break
        except json.JSONDecodeError:
            # If parsing fails, assume the assistant is asking for clarification
            user_input = input("You: ")
            conversation.append({"role": "user", "content": user_input})