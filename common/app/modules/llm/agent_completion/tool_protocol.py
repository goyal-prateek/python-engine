"""Structural protocol for tools passed into agent completions (no import from agents.tools)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from google.genai import types as gemini_types
from openai.types.chat import ChatCompletionToolParam
from pydantic import BaseModel


@runtime_checkable
class ToolSpec(Protocol):
    """Anything the completion router can serialize for OpenAI and Gemini."""

    name: str
    input_model: type[BaseModel]

    @property
    def description(self) -> str: ...

    def to_openai_tool(self) -> ChatCompletionToolParam: ...

    def to_gemini_tool(self) -> gemini_types.Tool: ...
