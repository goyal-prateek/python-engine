"""Completion backend interface (implemented per provider)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from common.app.modules.llm.agent_completion.request import AgentCompletionRequest
from common.app.modules.llm.agent_completion.stream_sink import AgentStreamSink
from common.app.modules.llm.messages import CompletionMessageModel


@runtime_checkable
class CompletionBackend(Protocol):
    async def complete(
        self,
        request: AgentCompletionRequest,
        *,
        stream_sink: AgentStreamSink | None = None,
    ) -> CompletionMessageModel: ...
