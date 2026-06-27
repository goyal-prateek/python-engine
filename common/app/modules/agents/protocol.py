"""Orchestration hooks between agents and the host (HITL, abort, streaming UX)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from common.app.modules.llm.messages import (
    ImageBlockParamModel,
    MessageParamModel,
    TextBlockParamModel,
)


class AgentQuestion(BaseModel):
    """Question for ask-user HITL flows."""

    id: str = Field(..., description="Stable question id")
    prompt: str = Field(..., description="Text shown to the user")
    kind: Literal["text", "single_select", "multi_select"] = Field(
        default="text",
        description="text = freeform; select kinds require options",
    )
    options: list[str] = Field(
        default_factory=list,
        description="Choices for single_select / multi_select",
    )

    @model_validator(mode="after")
    def _options_when_select(self) -> AgentQuestion:
        if self.kind != "text" and len(self.options) < 1:
            raise ValueError("single_select and multi_select require at least one option")
        return self


ConversationInjectBlock = TextBlockParamModel | ImageBlockParamModel


class CopilotProtocol(ABC):
    @abstractmethod
    async def should_continue_loop(self) -> bool:
        """Return False to stop before the next LLM call (after current tools finish)."""

    @abstractmethod
    async def mark_loop_as_completed(self) -> None:
        """Mark normal completion of the agent run."""

    @abstractmethod
    async def mark_loop_as_interrupted(self) -> None:
        """Mark abnormal exit (exception, shutdown)."""

    @abstractmethod
    async def mark_loop_as_in_progress(self) -> None:
        """Mark that a run has started."""

    @abstractmethod
    async def send_notification(self, update: MessageParamModel) -> None:
        """Fire-and-forget notification to the host (logging, websockets, etc.)."""

    @abstractmethod
    async def get_unhandled_messages(self) -> list[ConversationInjectBlock]:
        """Interrupt messages that may cancel in-flight tool batches."""

    async def get_background_context_messages(self) -> list[ConversationInjectBlock]:
        """Passive context injected at safe checkpoints (never cancels tools)."""
        return []

    @abstractmethod
    async def get_sticky_context(self) -> str | list[TextBlockParamModel]:
        """Optional sticky prefix merged into history."""

    def get_last_stream_event_at(self) -> float:
        return 0.0

    def reset_stream_event_at(self) -> None:
        """Hook for streaming watchdogs (unused when stream=False)."""

    async def on_stream_event(self) -> None:
        pass

    async def on_text_chunk(self, chunk: str) -> None:
        pass

    async def on_thinking_chunk(self, chunk: str) -> None:
        pass

    async def on_tool_use_start(self, tool_name: str, tool_id: str) -> None:
        pass

    async def on_tool_input_chunk(self, tool_id: str, chunk: str) -> None:
        pass

    async def on_tool_use_complete(self, tool_name: str, tool_id: str) -> None:
        pass

    async def on_tool_result(self, tool_id: str, result: str, is_error: bool = False) -> None:
        pass

    async def on_stream_start(self) -> None:
        pass

    async def on_stream_complete(self) -> None:
        pass

    async def on_compaction_start(self, compaction_id: str) -> None:
        pass

    async def on_compaction_complete(self, compaction_id: str, success: bool, summary: str) -> None:
        pass

    async def on_context_usage(self, used_tokens: int, total_tokens: int) -> None:
        pass

    async def on_question_request(
        self,
        tool_id: str,
        questions: list[AgentQuestion],
        context: str | None = None,
    ) -> None:
        pass

    async def on_permission_request(
        self,
        tool_id: str,
        title: str,
        body: str,
        context: str | None = None,
    ) -> None:
        """Host shows allow/cancel (optional custom message); resume via HITL tool result."""

        pass


class CopilotStreamSinkBridge:
    """Maps `CopilotProtocol` streaming hooks to `AgentStreamSink` for the LLM backends."""

    __slots__ = ("_p",)

    def __init__(self, p: CopilotProtocol) -> None:
        self._p = p

    async def on_stream_start(self) -> None:
        await self._p.on_stream_start()

    async def on_stream_complete(self) -> None:
        await self._p.on_stream_complete()

    async def on_stream_event(self) -> None:
        await self._p.on_stream_event()

    async def on_text_chunk(self, chunk: str) -> None:
        await self._p.on_text_chunk(chunk)

    async def on_thinking_chunk(self, chunk: str) -> None:
        await self._p.on_thinking_chunk(chunk)


class NullCopilotProtocol(CopilotProtocol):
    """Default no-op protocol for scripts and tests."""

    async def should_continue_loop(self) -> bool:
        return True

    async def mark_loop_as_completed(self) -> None:
        return None

    async def mark_loop_as_interrupted(self) -> None:
        return None

    async def mark_loop_as_in_progress(self) -> None:
        return None

    async def send_notification(self, update: MessageParamModel) -> None:
        return None

    async def get_unhandled_messages(self) -> list[ConversationInjectBlock]:
        return []

    async def get_sticky_context(self) -> str | list[TextBlockParamModel]:
        return ""
