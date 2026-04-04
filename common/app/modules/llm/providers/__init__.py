"""LLM facade: single entry for routed completions (agents and prompt-style flows)."""

from common.app.modules.llm.agent_completion import CompletionRouter
from common.app.modules.llm.agent_completion.request import AgentCompletionRequest
from common.app.modules.llm.clients import SharedLLMClients
from common.app.modules.llm.messages import CompletionMessageModel


class LLMProvider:
    @staticmethod
    async def complete(
        request: AgentCompletionRequest,
        shared_clients: SharedLLMClients,
    ) -> CompletionMessageModel:
        """Tool-capable completion via CompletionRouter (Gemini / OpenRouter)."""
        return await CompletionRouter(shared_clients).complete(request)
