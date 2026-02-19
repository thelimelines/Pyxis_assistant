from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

SCOPES = ["https://www.googleapis.com/auth/calendar"]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
TOKEN_FILE = PROJECT_ROOT / "token.json"
CREDENTIALS_FILE = PROJECT_ROOT / "credentials.json"
ENV_FILE = PROJECT_ROOT / ".env"
DEFAULT_CHAT_MODEL = "gpt-4o-mini"
DEFAULT_EVENT_MODEL = "gpt-5-mini"
DEFAULT_TRANSCRIBE_MODEL = "gpt-4o-mini-transcribe"
DEFAULT_TTS_BACKEND = "openai"
DEFAULT_TTS_MODEL = "gpt-4o-mini-tts"
DEFAULT_TTS_VOICE = "coral"
DEFAULT_TTS_SPEED = 1.0
DEFAULT_CUSTOM_WAKE_THRESHOLD = 0.60
DEFAULT_OPENWAKE_THRESHOLD = 0.35


def load_environment() -> None:
    load_dotenv(ENV_FILE)


def get_openai_api_key() -> str:
    load_environment()
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        try:
            from pyxis_assistant.profiles import get_profile_api_key

            profile = get_profile_api_key()
            if profile:
                key = profile.api_key
        except Exception:
            key = ""
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not set. Add it to .env.")
    return key


def set_openai_api_key(value: str) -> None:
    normalized = value.strip()
    if not normalized:
        msg = "OPENAI_API_KEY cannot be empty."
        raise ValueError(msg)

    lines: list[str] = []
    if ENV_FILE.exists():
        lines = ENV_FILE.read_text(encoding="utf-8").splitlines()

    updated = False
    output: list[str] = []
    for line in lines:
        if line.startswith("OPENAI_API_KEY="):
            output.append(f"OPENAI_API_KEY={normalized}")
            updated = True
        else:
            output.append(line)

    if not updated:
        output.append(f"OPENAI_API_KEY={normalized}")

    ENV_FILE.write_text("\n".join(output).strip() + "\n", encoding="utf-8")
    os.environ["OPENAI_API_KEY"] = normalized


def get_chat_model() -> str:
    load_environment()
    return os.getenv("PYXIS_CHAT_MODEL", DEFAULT_CHAT_MODEL).strip() or DEFAULT_CHAT_MODEL


def get_event_model() -> str:
    load_environment()
    return os.getenv("PYXIS_EVENT_MODEL", DEFAULT_EVENT_MODEL).strip() or DEFAULT_EVENT_MODEL


def get_transcribe_model() -> str:
    load_environment()
    return (
        os.getenv("PYXIS_TRANSCRIBE_MODEL", DEFAULT_TRANSCRIBE_MODEL).strip()
        or DEFAULT_TRANSCRIBE_MODEL
    )


def get_tts_backend() -> str:
    load_environment()
    backend = os.getenv("PYXIS_TTS_BACKEND", DEFAULT_TTS_BACKEND).strip().lower()
    if backend in {"openai", "windows", "pyttsx3", "auto"}:
        return backend
    return DEFAULT_TTS_BACKEND


def get_tts_model() -> str:
    load_environment()
    return os.getenv("PYXIS_TTS_MODEL", DEFAULT_TTS_MODEL).strip() or DEFAULT_TTS_MODEL


def get_tts_voice() -> str:
    load_environment()
    return os.getenv("PYXIS_TTS_VOICE", DEFAULT_TTS_VOICE).strip() or DEFAULT_TTS_VOICE


def get_tts_speed() -> float:
    load_environment()
    raw = os.getenv("PYXIS_TTS_SPEED", str(DEFAULT_TTS_SPEED)).strip()
    try:
        speed = float(raw)
    except ValueError:
        return DEFAULT_TTS_SPEED
    return min(4.0, max(0.25, speed))


def get_custom_wake_threshold() -> float:
    load_environment()
    raw = os.getenv("PYXIS_CUSTOM_WAKE_THRESHOLD", str(DEFAULT_CUSTOM_WAKE_THRESHOLD)).strip()
    try:
        value = float(raw)
    except ValueError:
        return DEFAULT_CUSTOM_WAKE_THRESHOLD
    return min(0.99, max(0.05, value))


def get_openwake_threshold() -> float:
    load_environment()
    raw = os.getenv("PYXIS_OPENWAKE_THRESHOLD", str(DEFAULT_OPENWAKE_THRESHOLD)).strip()
    try:
        value = float(raw)
    except ValueError:
        return DEFAULT_OPENWAKE_THRESHOLD
    return min(0.99, max(0.05, value))
