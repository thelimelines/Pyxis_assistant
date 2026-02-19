from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, cast

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

from pyxis_assistant.calendar import CalendarEvent, add_calendar_event, list_calendar_events
from pyxis_assistant.config import (
    PROJECT_ROOT,
    get_chat_model,
    get_event_model,
    get_openai_api_key,
)
from pyxis_assistant.logging_utils import get_logger

APP_DIR = PROJECT_ROOT / ".pyxis"
MEMORY_FILE = APP_DIR / "memory.json"
TASKS_FILE = APP_DIR / "tasks.json"
LOGGER = get_logger("pyxis.assistant")
_CLIENT: OpenAI | None = None


def _get_client() -> OpenAI:
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = OpenAI(api_key=get_openai_api_key())
    return _CLIENT


@dataclass
class AssistantResponse:
    text: str
    success: bool
    end_conversation: bool = False
    requires_event_confirmation: bool = False
    pending_event_json: str | None = None


def _ensure_store_dir() -> None:
    APP_DIR.mkdir(parents=True, exist_ok=True)


def _load_json(path: Path, default: Any) -> Any:
    _ensure_store_dir()
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default


def _save_json(path: Path, payload: Any) -> None:
    _ensure_store_dir()
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _load_memory() -> list[dict[str, str]]:
    payload = _load_json(MEMORY_FILE, [])
    if not isinstance(payload, list):
        return []
    return [item for item in payload if isinstance(item, dict)]


def _append_memory(role: str, content: str) -> None:
    history = _load_memory()
    history.append(
        {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
    )
    _save_json(MEMORY_FILE, history[-30:])


def _load_tasks() -> list[dict[str, Any]]:
    payload = _load_json(TASKS_FILE, [])
    if not isinstance(payload, list):
        return []
    return [item for item in payload if isinstance(item, dict)]


def _save_tasks(tasks: list[dict[str, Any]]) -> None:
    _save_json(TASKS_FILE, tasks)


def parse_with_actions(user_text: str) -> dict[str, Any]:
    memory = _load_memory()
    memory_text = "\n".join(
        f"{item.get('role', 'unknown')}: {item.get('content', '')}" for item in memory[-10:]
    )
    now_context = datetime.now().astimezone().isoformat(timespec="seconds")

    system_prompt = (
        "You are Pyxis, a personal assistant. "
        "Classify intent as one of: chat, add_event, add_task, list_tasks, list_events, "
        "end_conversation. "
        "Return ONLY JSON with keys: intent, reply, end_conversation, event(optional), "
        "task(optional). "
        "Set end_conversation=true if user wants to stop/finish the chat. "
        "For add_task include title and optional due,priority,notes,subtasks(list). "
        "Use ISO 8601 datetimes for event times where possible."
    )
    client = _get_client()
    response = client.chat.completions.create(
        model=get_chat_model(),
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "system",
                "content": f"Recent memory:\n{memory_text}" if memory_text else "No recent memory.",
            },
            {
                "role": "system",
                "content": f"Current local datetime is {now_context}.",
            },
            {"role": "user", "content": user_text},
        ],
    )
    content = response.choices[0].message.content or ""
    start = content.find("{")
    end = content.rfind("}") + 1
    if start == -1 or end <= 0:
        msg = "Assistant did not return valid JSON."
        raise ValueError(msg)

    payload = json.loads(content[start:end])
    if not isinstance(payload, dict):
        msg = "Assistant response was not an object."
        raise ValueError(msg)
    LOGGER.info("Intent payload: %s", payload)
    return payload


def _event_json_attempt(
    client: OpenAI,
    model: str,
    user_text: str,
    previous_json: str | None = None,
    previous_error: str | None = None,
) -> str:
    now_context = datetime.now().astimezone().isoformat(timespec="seconds")
    system_prompt = (
        "You generate calendar events as strict JSON only. "
        "Current local datetime is "
        f"{now_context}. "
        "Return one JSON object with keys: summary,start_time,end_time and optional "
        "time_zone,recurrence,description,location,reminder_minutes_before,duration_minutes. "
        "Use ISO 8601 datetime strings. "
        "For repeating schedules, use ONE event with recurrence RRULE string(s), "
        "never an array of event objects. "
        "Do not return dates in the past unless the user explicitly asks. "
        "No markdown, no prose."
    )
    messages: list[ChatCompletionMessageParam] = [
        cast(ChatCompletionMessageParam, {"role": "system", "content": system_prompt}),
        cast(
            ChatCompletionMessageParam,
            {"role": "user", "content": f"Create event JSON for: {user_text}"},
        ),
    ]
    if previous_json and previous_error:
        messages.append(
            cast(
                ChatCompletionMessageParam,
                {
                    "role": "user",
                    "content": (
                        "Previous JSON failed. Fix it.\n"
                        f"JSON: {previous_json}\n"
                        f"Error: {previous_error}"
                    ),
                },
            )
        )

    response = client.chat.completions.create(
        model=model,
        messages=messages,
    )
    content = response.choices[0].message.content or ""
    start = content.find("{")
    end = content.rfind("}") + 1
    if start == -1 or end <= 0:
        msg = "No JSON object returned for event."
        raise ValueError(msg)
    return content[start:end]


def _build_event_with_retries(
    user_text: str, max_attempts: int = 3
) -> tuple[bool, str, str | None]:
    client = _get_client()
    event_model = get_event_model()
    previous_json: str | None = None
    previous_error: str | None = None

    for _ in range(max_attempts):
        try:
            event_json = _event_json_attempt(
                client,
                event_model,
                user_text,
                previous_json,
                previous_error,
            )
            # Validate structure but do not write to calendar yet.
            CalendarEvent.from_json(event_json)
            return True, "Event parsed and ready for confirmation.", event_json
        except Exception as exc:
            previous_error = str(exc)
            previous_json = event_json if "event_json" in locals() else previous_json

    return False, str(previous_error or "Unknown event parsing error."), None


def finalize_event_from_json(event_json: str) -> tuple[bool, str]:
    try:
        event = CalendarEvent.from_json(event_json)
        link = add_calendar_event(event)
        LOGGER.info("Event confirmed and added. summary=%s", event.summary)
        return True, (link or "")
    except Exception as exc:
        LOGGER.warning("Event finalize failed: %s", exc)
        return False, str(exc)


def _handle_add_task(task_payload: dict[str, Any]) -> str:
    title = str(task_payload.get("title", "")).strip()
    if not title:
        msg = "Task title was missing."
        raise ValueError(msg)

    entry = {
        "title": title,
        "due": str(task_payload.get("due", "")).strip() or None,
        "priority": str(task_payload.get("priority", "")).strip() or None,
        "notes": str(task_payload.get("notes", "")).strip() or None,
        "subtasks": [
            str(item).strip() for item in task_payload.get("subtasks", []) if str(item).strip()
        ],
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "done": False,
    }
    tasks = _load_tasks()
    tasks.append(entry)
    _save_tasks(tasks)
    return f"Task added: {title}"


def _handle_list_tasks() -> str:
    tasks = _load_tasks()
    active = [task for task in tasks if not bool(task.get("done"))]
    if not active:
        return "You have no open tasks."
    lines = [
        f"{idx}. {task.get('title', 'Untitled task')}" for idx, task in enumerate(active, start=1)
    ]
    return "Open tasks:\n" + "\n".join(lines)


def _handle_list_events() -> str:
    events = list_calendar_events(max_results=5)
    if not events:
        return "No upcoming events found."
    lines = [f"{start}: {summary}" for start, summary in events]
    return "Upcoming events:\n" + "\n".join(lines)


def process_user_text(user_text: str) -> AssistantResponse:
    stripped = user_text.strip()
    if not stripped:
        return AssistantResponse(
            text="I did not hear anything.", success=False, end_conversation=False
        )

    payload = parse_with_actions(stripped)
    intent = str(payload.get("intent", "chat")).strip().lower()
    reply = str(payload.get("reply", "")).strip()
    end_conversation = bool(payload.get("end_conversation", False))
    try:
        if intent == "add_event":
            ok, detail, event_json = _build_event_with_retries(stripped, max_attempts=3)
            if ok:
                event = CalendarEvent.from_json(event_json or "{}")
                recurrence_text = (
                    ", ".join(event.recurrence) if event.recurrence else "None"
                )
                reminders_text = (
                    ", ".join(str(item) for item in event.reminder_minutes_before)
                    if event.reminder_minutes_before
                    else "Default"
                )
                output = (
                    "I understood this event:\n"
                    f"- Summary: {event.summary}\n"
                    f"- Start: {event.start_time}\n"
                    f"- End: {event.end_time}\n"
                    f"- Time zone: {event.time_zone}\n"
                    f"- Recurrence: {recurrence_text}\n"
                    f"- Reminder minutes before: {reminders_text}\n"
                    "Say 'yes' to confirm, or 'no' to cancel."
                )
                return AssistantResponse(
                    text=output,
                    success=True,
                    end_conversation=False,
                    requires_event_confirmation=True,
                    pending_event_json=event_json,
                )
            else:
                output = (
                    "I couldn't get that event right after three tries. "
                    "Let's keep chatting and try again differently."
                )
                return AssistantResponse(
                    text=output,
                    success=False,
                    end_conversation=False,
                )
        elif intent == "add_task":
            task_payload = payload.get("task")
            if not isinstance(task_payload, dict):
                raise ValueError("Task details missing.")
            output = _handle_add_task(task_payload)
        elif intent == "list_tasks":
            output = _handle_list_tasks()
        elif intent == "list_events":
            output = _handle_list_events()
        elif intent == "end_conversation":
            output = reply or "Okay, stopping here. Say pyxis when you need me."
            end_conversation = True
        else:
            output = reply or "Okay."
    except Exception as exc:
        LOGGER.warning("process_user_text failed: %s", exc)
        return AssistantResponse(text=str(exc), success=False, end_conversation=False)

    _append_memory("user", stripped)
    _append_memory("assistant", output)
    return AssistantResponse(text=output, success=True, end_conversation=end_conversation)
