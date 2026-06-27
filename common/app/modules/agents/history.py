"""In-memory conversation history with token bookkeeping (best-effort)."""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from common.app.modules.llm.messages import (
    MessageParamModel,
    TextBlockParamModel,
    ToolResultBlockParamModel,
    ToolResultTextBlock,
    Usage,
)


class MessageHistory:
    def __init__(
        self,
        *,
        context_window_tokens: int,
        session_id: str,
        refresh_sticky_context: Callable[[], Awaitable[str | list[TextBlockParamModel]]],
    ) -> None:
        self.context_window_tokens = context_window_tokens
        self.session_id = session_id
        self.refresh_sticky_context = refresh_sticky_context
        self.messages: list[MessageParamModel] = []
        self.total_tokens = 0
        self.message_tokens: list[tuple[int, int]] = []

    async def bootstrap_with_sticky(self) -> None:
        sticky = await self.refresh_sticky_context()
        blocks: list[TextBlockParamModel]
        if isinstance(sticky, str):
            blocks = [TextBlockParamModel(type="text", text=sticky)] if sticky.strip() else []
        else:
            blocks = [b for b in sticky if b.text.strip()]
        if blocks:
            self.messages.append(
                MessageParamModel(role="user", content=list(blocks)),
            )

    def format_for_api(self) -> list[MessageParamModel]:
        return list(self.messages)

    async def add_message(
        self,
        message: MessageParamModel,
        usage: Usage | None = None,
    ) -> None:
        self.messages.append(message)
        if message.role == "assistant" and usage is not None:
            total_in = (
                usage.input_tokens
                + usage.cache_read_input_tokens
                + usage.cache_creation_input_tokens
            )
            out = usage.output_tokens
            current_turn_in = total_in - self.total_tokens
            self.message_tokens.append((current_turn_in, out))
            self.total_tokens += current_turn_in + out

    async def add_human_in_the_loop_message(self, tool_id: str, tool_result: str) -> None:
        for message in self.messages:
            for content in message.content:
                if isinstance(content, ToolResultBlockParamModel) and (
                    content.tool_use_id == tool_id
                ):
                    if isinstance(content.content, str):
                        content.content += "\n\n" + tool_result
                    else:
                        content.content.append(
                            ToolResultTextBlock(type="text", text="\n\n" + tool_result)
                        )
                    return
        self.messages.append(
            MessageParamModel(
                role="user",
                content=[
                    TextBlockParamModel(
                        type="text",
                        text=f"[human_in_the_loop tool={tool_id}]\n{tool_result}",
                    )
                ],
            )
        )

    def truncate(self) -> None:
        """Drop oldest turns until under context budget (best-effort)."""
        while self.total_tokens > self.context_window_tokens and len(self.messages) > 2:
            self.messages.pop(0)
            if self.message_tokens:
                self.message_tokens.pop(0)
            self.total_tokens = max(0, self.total_tokens - 1000)
