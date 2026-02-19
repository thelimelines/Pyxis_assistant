from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

from pyxis_assistant.config import PROJECT_ROOT

APP_DIR = PROJECT_ROOT / ".pyxis"
PROFILES_FILE = APP_DIR / "profiles.json"


@dataclass
class ProfileSelection:
    name: str
    api_key: str


def _default_store() -> dict[str, Any]:
    return {
        "default_profile": "default",
        "profiles": {},
    }


def _ensure_app_dir() -> None:
    APP_DIR.mkdir(parents=True, exist_ok=True)


def load_store() -> dict[str, Any]:
    _ensure_app_dir()
    if not PROFILES_FILE.exists():
        return _default_store()

    try:
        data = json.loads(PROFILES_FILE.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return _default_store()
        if "profiles" not in data or not isinstance(data["profiles"], dict):
            data["profiles"] = {}
        if "default_profile" not in data or not isinstance(data["default_profile"], str):
            data["default_profile"] = "default"
        return data
    except json.JSONDecodeError:
        return _default_store()


def save_store(store: dict[str, Any]) -> None:
    _ensure_app_dir()
    PROFILES_FILE.write_text(json.dumps(store, indent=2), encoding="utf-8")


def list_profile_names() -> list[str]:
    store = load_store()
    names = sorted(store["profiles"].keys())
    return names if names else ["default"]


def default_profile_name() -> str:
    store = load_store()
    return str(store.get("default_profile", "default"))


def set_profile_api_key(profile_name: str, api_key: str, make_default: bool = True) -> None:
    normalized_profile = profile_name.strip() or "default"
    normalized_key = api_key.strip()
    if not normalized_key:
        msg = "API key cannot be empty."
        raise ValueError(msg)

    store = load_store()
    profiles = store["profiles"]
    profiles[normalized_profile] = {"openai_api_key": normalized_key}
    if make_default:
        store["default_profile"] = normalized_profile

    save_store(store)
    os.environ["OPENAI_API_KEY"] = normalized_key


def get_profile_api_key(profile_name: str | None = None) -> ProfileSelection | None:
    store = load_store()
    selected = (profile_name or str(store.get("default_profile", "default"))).strip() or "default"
    details = store["profiles"].get(selected)
    if not isinstance(details, dict):
        return None

    key = str(details.get("openai_api_key", "")).strip()
    if not key:
        return None

    os.environ["OPENAI_API_KEY"] = key
    return ProfileSelection(name=selected, api_key=key)
