from __future__ import annotations

import json
from pathlib import Path

from openai import OpenAI

from pyxis_assistant.config import get_event_model, get_openai_api_key, get_transcribe_model

SYSTEM_PROMPT = (
    "You are a calendar event parser. "
    "Return only a single JSON object with keys: summary, start_time, end_time. "
    "Optional keys: time_zone, recurrence. "
    "Use ISO 8601 format for start_time and end_time. "
    "Do not include markdown fences or explanation text."
)
_CLIENT: OpenAI | None = None


def _get_client() -> OpenAI:
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = OpenAI(api_key=get_openai_api_key())
    return _CLIENT


def extract_event_json(user_input: str) -> str:
    client = _get_client()
    response = client.chat.completions.create(
        model=get_event_model(),
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_input},
        ],
    )
    content = response.choices[0].message.content
    if not content:
        msg = "OpenAI returned an empty response."
        raise ValueError(msg)

    payload = _slice_json_object(content)
    json.loads(payload)
    return payload


def transcribe_audio(path: Path) -> str:
    client = _get_client()
    with path.open("rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model=get_transcribe_model(),
            file=audio_file,
        )

    text = transcript.text.strip()
    if not text:
        msg = "Transcription returned empty text."
        raise ValueError(msg)
    return text


def _slice_json_object(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end <= 0:
        msg = "No JSON object found in model response."
        raise ValueError(msg)
    return text[start:end]
