"""Canonical LLM message types for multi-provider agent completions."""

from common.app.modules.llm.messages.blocks import (
    ImageBlockParamModel,
    MessageContentBlock,
    MessageParamModel,
    TextBlockParamModel,
    ThinkingBlockParamModel,
    ToolResultBlockParamModel,
    ToolResultContentBlock,
    ToolResultImageBlock,
    ToolResultTextBlock,
    ToolUseBlockParamModel,
    flatten_messages_to_openai,
    messages_from_llm_prompt_items,
)
from common.app.modules.llm.messages.completion import (
    CompletionMessageModel,
    StopReason,
    Usage,
)

__all__ = [
    "CompletionMessageModel",
    "ImageBlockParamModel",
    "MessageContentBlock",
    "MessageParamModel",
    "StopReason",
    "TextBlockParamModel",
    "ThinkingBlockParamModel",
    "ToolResultBlockParamModel",
    "ToolResultContentBlock",
    "ToolResultImageBlock",
    "ToolResultTextBlock",
    "ToolUseBlockParamModel",
    "Usage",
    "flatten_messages_to_openai",
    "messages_from_llm_prompt_items",
]
