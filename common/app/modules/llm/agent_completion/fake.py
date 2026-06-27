"""Test double for CompletionBackend."""

from __future__ import annotations

from collections import deque
from typing import Any

from common.app.modules.llm.agent_completion.request import AgentCompletionRequest
from common.app.modules.llm.messages import CompletionMessageModel


class FakeCompletionBackend:
    """Returns queued completions in order (raises if empty)."""

    def __init__(self, responses: deque[CompletionMessageModel] | None = None) -> None:
        self._queue: deque[CompletionMessageModel] = responses or deque()

    def enqueue(self, response: CompletionMessageModel) -> None:
        self._queue.append(response)

    async def complete(
        self,
        request: AgentCompletionRequest,
        *,
        stream_sink: Any = None,
    ) -> CompletionMessageModel:
        if stream_sink is not None:
            raise NotImplementedError("FakeCompletionBackend does not support streaming")
        if not self._queue:
            raise RuntimeError("FakeCompletionBackend: no queued responses")
        return self._queue.popleft()
