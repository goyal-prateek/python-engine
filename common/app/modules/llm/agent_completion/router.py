"""Routes agent completions by AgentModelConfig.provider with optional fallbacks."""

from __future__ import annotations

from common.app.modules.llm.agent_completion.gemini_backend import GeminiAgentBackend
from common.app.modules.llm.agent_completion.openai_backend import OpenAICompatibleAgentBackend
from common.app.modules.llm.agent_completion.request import (
    AgentCompletionRequest,
    AgentModelConfig,
)
from common.app.modules.llm.agent_completion.stream_sink import AgentStreamSink
from common.app.modules.llm.clients import SharedLLMClients
from common.app.modules.llm.messages import CompletionMessageModel


class CompletionRouter:
    """Dispatches to Gemini or OpenAI-compatible backends; Anthropic reserved."""

    def __init__(self, clients: SharedLLMClients) -> None:
        self._gemini = GeminiAgentBackend(clients.gemini_client)
        self._openai = OpenAICompatibleAgentBackend(clients.openrouter_openai)

    async def complete(
        self,
        request: AgentCompletionRequest,
        *,
        stream_sink: AgentStreamSink | None = None,
    ) -> CompletionMessageModel:
        configs: list[AgentModelConfig] = [request.config, *request.config.fallback_models]
        last_error: Exception | None = None
        for cfg in configs:
            sub = AgentCompletionRequest(
                messages=request.messages,
                system=request.system,
                tools=request.tools,
                config=cfg,
            )
            try:
                return await self._dispatch(sub, stream_sink=stream_sink)
            except Exception as e:
                last_error = e
                continue
        if last_error is not None:
            raise last_error
        raise RuntimeError("No model configuration available for completion")

    async def _dispatch(
        self,
        request: AgentCompletionRequest,
        *,
        stream_sink: AgentStreamSink | None = None,
    ) -> CompletionMessageModel:
        match request.config.provider:
            case "gemini":
                return await self._gemini.complete(request, stream_sink=stream_sink)
            case "openai":
                return await self._openai.complete(request, stream_sink=stream_sink)
            case "anthropic":
                raise NotImplementedError(
                    "Anthropic Claude is not implemented; use provider 'gemini' or 'openai'."
                )
