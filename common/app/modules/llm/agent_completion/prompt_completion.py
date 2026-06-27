"""Single-turn completions from LLMPromptItem lists via CompletionRouter."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Sequence

from common.app.modules.llm.agent_completion.request import (
    AgentCompletionRequest,
    AgentModelConfig,
)
from common.app.modules.llm.agent_completion.router import CompletionRouter
from common.app.modules.llm.agent_completion.stream_sink import AgentStreamSink
from common.app.modules.llm.clients import SharedLLMClients
from common.app.modules.llm.messages import CompletionMessageModel
from common.app.modules.llm.promtps import LLMPromptItem


class _QueuedTextStreamSink:
    """Forwards OpenAI-compatible stream text deltas into an asyncio queue."""

    __slots__ = ("_queue",)

    def __init__(self, queue: asyncio.Queue[str | None]) -> None:
        self._queue = queue

    async def on_stream_start(self) -> None:
        return None

    async def on_stream_complete(self) -> None:
        return None

    async def on_stream_event(self) -> None:
        return None

    async def on_text_chunk(self, chunk: str) -> None:
        await self._queue.put(chunk)

    async def on_thinking_chunk(self, chunk: str) -> None:
        return None


async def complete_llm_prompt_items(
    clients: SharedLLMClients,
    *,
    prompt: Sequence[LLMPromptItem],
    system: str = "",
    config: AgentModelConfig,
) -> CompletionMessageModel:
    router = CompletionRouter(clients)
    request = AgentCompletionRequest.from_llm_prompt_items(prompt, system=system, config=config)
    return await router.complete(request)


async def stream_llm_prompt_text_chunks(
    clients: SharedLLMClients,
    *,
    prompt: Sequence[LLMPromptItem],
    system: str = "",
    config: AgentModelConfig,
) -> AsyncIterator[str]:
    if config.provider != "openai":
        raise NotImplementedError(
            "Streaming prompt completions are only implemented for provider 'openai' "
            "(e.g. OpenRouter). Use stream=False for Gemini."
        )
    queue: asyncio.Queue[str | None] = asyncio.Queue()
    sink: AgentStreamSink = _QueuedTextStreamSink(queue)
    router = CompletionRouter(clients)
    request = AgentCompletionRequest.from_llm_prompt_items(prompt, system=system, config=config)

    async def run_completion() -> None:
        try:
            await router.complete(request, stream_sink=sink)
        finally:
            await queue.put(None)

    task = asyncio.create_task(run_completion())
    try:
        while True:
            item = await queue.get()
            if item is None:
                break
            yield item
    finally:
        await task
