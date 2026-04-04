"""Typed request envelope for a single agent LLM turn."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Literal, Sequence

from pydantic import BaseModel, Field

from common.app.modules.llm.agent_completion.tool_protocol import ToolSpec
from common.app.modules.llm.messages import MessageParamModel, messages_from_llm_prompt_items
from common.app.modules.llm.promtps import LLMPromptItem


AgentProvider = Literal["gemini", "openai", "anthropic"]


class AgentModelConfig(BaseModel):
    """Per-agent model and provider selection."""

    provider: AgentProvider
    model: str
    max_tokens: int = 8192
    temperature: float = 0.7
    context_window_tokens: int = 1_000_000
    thinking_budget_tokens: int = 0
    preferred_key_cache_id: str = "default"
    fallback_models: List[AgentModelConfig] = Field(default_factory=list)
    #: Merged into OpenAI `extra_body` for OpenRouter (e.g. `{"reasoning": {"max_tokens": 4096}}`).
    openrouter_extra_body: dict[str, Any] = Field(default_factory=dict)


@dataclass(frozen=True)
class AgentCompletionRequest:
    """One non-streaming completion call (messages + system + tools + config)."""

    messages: tuple[MessageParamModel, ...]
    system: str
    tools: tuple[ToolSpec, ...]
    config: AgentModelConfig

    @classmethod
    def from_parts(
        cls,
        messages: Sequence[MessageParamModel],
        system: str,
        tools: Sequence[ToolSpec],
        config: AgentModelConfig,
    ) -> AgentCompletionRequest:
        return cls(
            messages=tuple(messages),
            system=system,
            tools=tuple(tools),
            config=config,
        )

    @classmethod
    def from_llm_prompt_items(
        cls,
        prompt: Sequence[LLMPromptItem],
        *,
        system: str,
        config: AgentModelConfig,
    ) -> AgentCompletionRequest:
        return cls(
            messages=tuple(messages_from_llm_prompt_items(prompt)),
            system=system,
            tools=(),
            config=config,
        )
