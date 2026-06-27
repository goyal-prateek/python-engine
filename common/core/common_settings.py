"""Settings that `common` libraries need (keys, AWS, etc.). Apps extend this with their own fields."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import find_dotenv, load_dotenv

_dotenv_loaded = False


def ensure_dotenv_loaded() -> None:
    global _dotenv_loaded
    if _dotenv_loaded:
        return
    path = find_dotenv()
    if path:
        load_dotenv()
    _dotenv_loaded = True


@dataclass
class CommonServiceSettings:
    OPENAI_API_KEY: str | None
    OPENROUTER_API_KEY: str | None
    GOOGLE_API_KEY: str | None
    GOOGLE_PROJECT_ID: str | None
    AWS_REGION: str
    AWS_ACCESS_KEY: str | None
    AWS_SECRET_ACCESS_KEY: str | None
    SMALLEST_AI_API_KEY: str | None
    MONGO_URI: str | None
    MONGO_DB_NAME: str


def common_settings_from_env() -> CommonServiceSettings:
    """Default for scripts/tests that use `common` without an app `configure()` call."""
    ensure_dotenv_loaded()
    return CommonServiceSettings(
        OPENAI_API_KEY=os.getenv("OPENAI_API_KEY"),
        OPENROUTER_API_KEY=os.getenv("OPENROUTER_API_KEY"),
        GOOGLE_API_KEY=os.getenv("GOOGLE_API_KEY"),
        GOOGLE_PROJECT_ID=os.getenv("GOOGLE_PROJECT_ID"),
        AWS_REGION=os.getenv("AWS_REGION") or "ap-south-1",
        AWS_ACCESS_KEY=os.getenv("AWS_ACCESS_KEY"),
        AWS_SECRET_ACCESS_KEY=os.getenv("AWS_SECRET_ACCESS_KEY"),
        SMALLEST_AI_API_KEY=os.getenv("SMALLEST_AI_API_KEY"),
        MONGO_URI=os.getenv("MONGO_URI"),
        MONGO_DB_NAME=os.getenv("MONGO_DB_NAME") or "python_engine",
    )
