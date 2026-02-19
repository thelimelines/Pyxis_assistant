from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

from pyxis_assistant.config import CREDENTIALS_FILE, SCOPES, TOKEN_FILE


@dataclass
class CalendarEvent:
    summary: str
    start_time: str
    end_time: str
    time_zone: str = "UTC"
    recurrence: list[str] | None = None
    description: str | None = None
    location: str | None = None
    reminder_minutes_before: list[int] | None = None

    @classmethod
    def from_json(cls, payload: str) -> CalendarEvent:
        data = json.loads(payload)
        if not isinstance(data, dict):
            msg = "Event payload must be a JSON object."
            raise ValueError(msg)

        required = ["summary", "start_time"]
        missing = [key for key in required if key not in data]
        if missing:
            msg = f"Missing required keys: {', '.join(missing)}"
            raise ValueError(msg)

        start_time = str(data["start_time"])
        end_time_value = data.get("end_time")
        duration_minutes = data.get("duration_minutes")
        if not end_time_value:
            if duration_minutes is None:
                msg = "Missing required key: end_time (or provide duration_minutes)."
                raise ValueError(msg)
            end_time_value = _derive_end_time(start_time, duration_minutes)

        recurrence = _normalize_recurrence(data.get("recurrence"))
        reminders = _normalize_reminders(data.get("reminder_minutes_before"))
        _validate_start_not_past(start_time, allow_past=bool(data.get("allow_past", False)))

        return cls(
            summary=str(data["summary"]),
            start_time=start_time,
            end_time=str(end_time_value),
            time_zone=str(data.get("time_zone", "UTC")),
            recurrence=recurrence,
            description=str(data["description"]).strip() if "description" in data else None,
            location=str(data["location"]).strip() if "location" in data else None,
            reminder_minutes_before=reminders,
        )


def _parse_iso(value: str) -> datetime:
    normalized = value.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed


def _derive_end_time(start_time: str, duration_minutes_raw: Any) -> str:
    try:
        duration_minutes = int(duration_minutes_raw)
    except (TypeError, ValueError):
        msg = "duration_minutes must be an integer."
        raise ValueError(msg) from None
    if duration_minutes <= 0:
        msg = "duration_minutes must be greater than zero."
        raise ValueError(msg)
    start_dt = _parse_iso(start_time)
    end_dt = start_dt + timedelta(minutes=duration_minutes)
    return end_dt.isoformat()


def _normalize_recurrence(raw: Any) -> list[str] | None:
    if raw is None:
        return None
    values: list[str]
    if isinstance(raw, str):
        values = [raw.strip()]
    elif isinstance(raw, list):
        values = [str(item).strip() for item in raw if str(item).strip()]
    else:
        values = [str(raw).strip()]

    filtered = [item for item in values if item]
    if not filtered:
        return None
    return filtered


def _normalize_reminders(raw: Any) -> list[int] | None:
    if raw is None:
        return None
    if not isinstance(raw, list):
        raw = [raw]

    values: list[int] = []
    for item in raw:
        try:
            minute = int(item)
        except (TypeError, ValueError):
            continue
        if minute >= 0:
            values.append(minute)
    if not values:
        return None
    # Keep unique and stable.
    return sorted(set(values))


def _validate_start_not_past(start_time: str, allow_past: bool) -> None:
    if allow_past:
        return
    start_dt = _parse_iso(start_time)
    now = datetime.now(UTC)
    if start_dt < now - timedelta(minutes=5):
        msg = (
            "Event start_time is in the past. Use a future date/time, "
            "or set allow_past=true explicitly."
        )
        raise ValueError(msg)


def authenticate_google_calendar() -> None:
    if not CREDENTIALS_FILE.exists():
        msg = f"Missing credentials file: {CREDENTIALS_FILE}"
        raise FileNotFoundError(msg)

    flow = InstalledAppFlow.from_client_secrets_file(str(CREDENTIALS_FILE), SCOPES)
    creds = flow.run_local_server(port=0, open_browser=True)
    TOKEN_FILE.write_text(creds.to_json(), encoding="utf-8")
    print(f"Authentication successful. Token saved to '{TOKEN_FILE.name}'.")


def ensure_calendar_authenticated() -> None:
    if TOKEN_FILE.exists():
        return

    if not CREDENTIALS_FILE.exists():
        msg = (
            f"Missing '{CREDENTIALS_FILE.name}'. Add your Google OAuth desktop credentials file "
            f"at: {CREDENTIALS_FILE}"
        )
        raise FileNotFoundError(msg)

    authenticate_google_calendar()


def _get_calendar_service() -> Any:
    ensure_calendar_authenticated()

    creds = Credentials.from_authorized_user_file(str(TOKEN_FILE), SCOPES)
    return build("calendar", "v3", credentials=creds)


def add_calendar_event(event: CalendarEvent) -> str | None:
    service = _get_calendar_service()
    event_payload = {
        "summary": event.summary,
        "start": {
            "dateTime": event.start_time,
            "timeZone": event.time_zone,
        },
        "end": {
            "dateTime": event.end_time,
            "timeZone": event.time_zone,
        },
    }

    if event.recurrence:
        event_payload["recurrence"] = event.recurrence
    if event.description:
        event_payload["description"] = event.description
    if event.location:
        event_payload["location"] = event.location
    if event.reminder_minutes_before:
        event_payload["reminders"] = {
            "useDefault": False,
            "overrides": [
                {"method": "popup", "minutes": minutes}
                for minutes in event.reminder_minutes_before
            ],
        }

    created_event = service.events().insert(calendarId="primary", body=event_payload).execute()
    return created_event.get("htmlLink")


def list_calendar_events(max_results: int = 10) -> list[tuple[str, str]]:
    service = _get_calendar_service()
    now = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    events_result = (
        service.events()
        .list(
            calendarId="primary",
            timeMin=now,
            maxResults=max_results,
            singleEvents=True,
            orderBy="startTime",
        )
        .execute()
    )
    events = events_result.get("items", [])

    output: list[tuple[str, str]] = []
    for event in events:
        start = event.get("start", {}).get("dateTime", event.get("start", {}).get("date", ""))
        summary = event.get("summary", "(No title)")
        output.append((str(start), str(summary)))

    return output
