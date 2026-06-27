"""Google Gemini generate_content backend for tool use."""

from __future__ import annotations

from typing import Any, cast

from google import genai
from google.genai.types import GenerateContentConfig, ThinkingConfig

from common.app.modules.llm.agent_completion.request import AgentCompletionRequest
from common.app.modules.llm.agent_completion.stream_sink import AgentStreamSink
from common.app.modules.llm.messages import CompletionMessageModel


class GeminiAgentBackend:
    def __init__(self, client: genai.Client) -> None:
        self._client = client

    async def complete(
        self,
        request: AgentCompletionRequest,
        *,
        stream_sink: AgentStreamSink | None = None,
    ) -> CompletionMessageModel:
        if stream_sink is not None:
            raise NotImplementedError(
                "Streaming agent completions for Gemini are not implemented; "
                "use provider 'openai' (e.g. OpenRouter) or pass stream=False on Agent."
            )
        contents = [m.to_gemini_message() for m in request.messages]
        tool_list = [t.to_gemini_tool() for t in request.tools]
        cfg = request.config
        thinking = (
            ThinkingConfig(include_thoughts=True, thinking_budget=cfg.thinking_budget_tokens)
            if cfg.thinking_budget_tokens >= 1024
            else ThinkingConfig(include_thoughts=False, thinking_budget=0)
        )
        gen_cfg = GenerateContentConfig(
            system_instruction=request.system or None,
            tools=cast(Any, tool_list if tool_list else None),
            temperature=cfg.temperature,
            max_output_tokens=cfg.max_tokens,
            thinking_config=thinking,
        )
        response = await self._client.aio.models.generate_content(
            model=cfg.model,
            contents=cast(Any, list(contents)),
            config=gen_cfg,
        )
        return CompletionMessageModel.from_gemini_message(response)
