"""Hooks for token/chunk streaming during agent LLM calls (mirrors CopilotProtocol)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class AgentStreamSink(Protocol):
    """Called by OpenAI-compatible backends while `stream=True` (e.g. OpenRouter)."""

    async def on_stream_start(self) -> None: ...

    async def on_stream_complete(self) -> None: ...

    async def on_stream_event(self) -> None: ...

    async def on_text_chunk(self, chunk: str) -> None: ...

    async def on_thinking_chunk(self, chunk: str) -> None: ...
