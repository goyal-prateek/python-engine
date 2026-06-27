"""OpenAI Chat Completions backend (OpenRouter, OpenAI, etc.)."""

from __future__ import annotations

from typing import Any

from openai import AsyncOpenAI
from openai._types import NOT_GIVEN

from common.app.modules.llm.agent_completion.openai_chat_stream import accumulate_openai_chat_stream
from common.app.modules.llm.agent_completion.request import AgentCompletionRequest
from common.app.modules.llm.agent_completion.stream_sink import AgentStreamSink
from common.app.modules.llm.messages import CompletionMessageModel, flatten_messages_to_openai


class OpenAICompatibleAgentBackend:
    def __init__(self, client: AsyncOpenAI) -> None:
        self._client = client

    def _create_kwargs(self, request: AgentCompletionRequest) -> dict[str, Any]:
        flat = flatten_messages_to_openai(list(request.messages))
        if request.system.strip():
            flat.insert(0, {"role": "system", "content": request.system})
        tools = [t.to_openai_tool() for t in request.tools]
        openai_tools = tools if tools else NOT_GIVEN
        cfg = request.config
        extra: dict[str, Any] = dict(cfg.openrouter_extra_body) if cfg.openrouter_extra_body else {}
        return {
            "model": cfg.model,
            "messages": flat,
            "tools": openai_tools,
            "max_tokens": cfg.max_tokens,
            "temperature": cfg.temperature,
            "extra_body": extra or NOT_GIVEN,
        }

    async def complete(
        self,
        request: AgentCompletionRequest,
        *,
        stream_sink: AgentStreamSink | None = None,
    ) -> CompletionMessageModel:
        base = self._create_kwargs(request)
        if stream_sink is not None:
            extra = base.get("extra_body")
            merged: dict[str, Any] = {}
            if extra is not NOT_GIVEN and isinstance(extra, dict):
                merged.update(extra)
            so = merged.get("stream_options")
            if isinstance(so, dict):
                stream_opts = {**so, "include_usage": True}
            else:
                stream_opts = {"include_usage": True}
            merged["stream_options"] = stream_opts
            stream_kwargs = {
                **{k: v for k, v in base.items() if k != "extra_body"},
                "stream": True,
                "extra_body": merged or NOT_GIVEN,
            }
            return await accumulate_openai_chat_stream(
                self._client,
                create_kwargs=stream_kwargs,
                stream_sink=stream_sink,
                fallback_model=request.config.model,
            )

        resp = await self._client.chat.completions.create(
            **base,
            stream=False,
        )
        return CompletionMessageModel.from_openai_chat_completion(resp)
