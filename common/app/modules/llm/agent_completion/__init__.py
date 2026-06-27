"""Multi-provider completions for agents (non-streaming and OpenAI-compatible streaming)."""

from common.app.modules.llm.agent_completion.backend_protocol import CompletionBackend
from common.app.modules.llm.agent_completion.fake import FakeCompletionBackend
from common.app.modules.llm.agent_completion.prompt_completion import (
    complete_llm_prompt_items,
    stream_llm_prompt_text_chunks,
)
from common.app.modules.llm.agent_completion.request import (
    AgentCompletionRequest,
    AgentModelConfig,
    AgentProvider,
)
from common.app.modules.llm.agent_completion.router import CompletionRouter
from common.app.modules.llm.agent_completion.stream_sink import AgentStreamSink
from common.app.modules.llm.agent_completion.tool_protocol import ToolSpec

__all__ = [
    "AgentCompletionRequest",
    "AgentModelConfig",
    "AgentProvider",
    "AgentStreamSink",
    "CompletionBackend",
    "CompletionRouter",
    "FakeCompletionBackend",
    "ToolSpec",
    "complete_llm_prompt_items",
    "stream_llm_prompt_text_chunks",
]
