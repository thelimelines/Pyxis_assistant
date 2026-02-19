from __future__ import annotations

import argparse
import getpass

from pyxis_assistant.calendar import (
    CalendarEvent,
    add_calendar_event,
    authenticate_google_calendar,
    list_calendar_events,
)
from pyxis_assistant.config import get_openai_api_key
from pyxis_assistant.gui import launch_gui
from pyxis_assistant.nlp import extract_event_json
from pyxis_assistant.profiles import set_profile_api_key


def main() -> int:
    parser = argparse.ArgumentParser(description="Pyxis Google Calendar assistant.")
    parser.add_argument(
        "--profile",
        help="API key profile name to use.",
    )
    parser.add_argument(
        "--set-api-key",
        action="store_true",
        help="Prompt to (re)define the API key profile before running.",
    )
    subparsers = parser.add_subparsers(dest="command", required=False)

    subparsers.add_parser("gui", help="Launch the desktop popup assistant.")
    subparsers.add_parser("auth", help="Authenticate with Google Calendar and create token.json.")
    subparsers.add_parser("list", help="List upcoming events.")

    add_parser = subparsers.add_parser("add", help="Add an event using OpenAI parsing.")
    add_parser.add_argument(
        "-p",
        "--prompt",
        help="Natural language event description. If omitted, interactive input is used.",
    )

    args = parser.parse_args()

    if args.set_api_key and args.command != "gui":
        profile_name = (args.profile or "default").strip() or "default"
        entered = getpass.getpass(f"Enter OpenAI API key for profile '{profile_name}': ").strip()
        if not entered:
            print("No API key entered.")
            return 1
        set_profile_api_key(profile_name=profile_name, api_key=entered, make_default=True)

    has_key = True
    try:
        get_openai_api_key()
    except RuntimeError:
        has_key = False

    if not has_key and args.command == "add" and not args.set_api_key:
        print(
            "No API key configured. Use `uv run pyxis --set-api-key` or set OPENAI_API_KEY in .env."
        )
        return 1

    if args.command in (None, "gui"):
        return launch_gui(
            force_redefine_api_key=args.set_api_key,
            preferred_profile=args.profile,
        )

    if args.command == "auth":
        authenticate_google_calendar()
        return 0

    if args.command == "list":
        events = list_calendar_events()
        if not events:
            print("No upcoming events found.")
            return 0
        print("Upcoming events:")
        for start, summary in events:
            print(f"- {start}: {summary}")
        return 0

    if args.command == "add":
        prompt = args.prompt or input("Describe the event you'd like to add:\n> ").strip()
        if not prompt:
            print("No event description provided.")
            return 1

        event_json = extract_event_json(prompt)
        event = CalendarEvent.from_json(event_json)
        link = add_calendar_event(event)
        print(f"Event created: {link or '(no link returned)'}")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
