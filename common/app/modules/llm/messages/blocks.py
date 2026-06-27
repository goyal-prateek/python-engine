"""Provider-agnostic chat blocks and message params (OpenAI + Gemini serialization)."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Literal

from google.genai import types as gemini_types
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionContentPartImageParam,
    ChatCompletionContentPartParam,
    ChatCompletionContentPartTextParam,
    ChatCompletionMessageParam,
    ChatCompletionMessageToolCallParam,
    ChatCompletionToolMessageParam,
    ChatCompletionUserMessageParam,
)
from openai.types.chat.chat_completion_message_tool_call_param import Function
from pydantic import BaseModel, Field

from common.app.modules.llm.promtps import LLMPromptItem


def _assistant_text_parts_to_openai_content(
    parts: list[ChatCompletionContentPartTextParam],
) -> str | None:
    """OpenAI assistant messages use a single string for `content`, not a parts list."""
    if not parts:
        return None
    return "\n\n".join(p["text"] for p in parts)


class ToolResultTextBlock(BaseModel):
    type: Literal["text"] = "text"
    text: str


class ToolResultImageBlock(BaseModel):
    class URLSource(BaseModel):
        type: Literal["url"] = "url"
        url: str

    type: Literal["image"] = "image"
    source: URLSource


ToolResultContentBlock = ToolResultTextBlock | ToolResultImageBlock


class ToolResultBlockParamModel(BaseModel):
    tool_use_id: str
    type: Literal["tool_result"] = "tool_result"
    content: str | list[ToolResultContentBlock]
    is_error: bool
    human_in_the_loop: bool = False
    tool_name: str

    def openai_type(self) -> ChatCompletionToolMessageParam:
        if isinstance(self.content, str):
            text_content = self.content
        else:
            text_parts: list[str] = []
            for block in self.content:
                if isinstance(block, ToolResultTextBlock):
                    text_parts.append(block.text)
                elif isinstance(block, ToolResultImageBlock):
                    text_parts.append(f"[Image: {block.source.url}]")
            text_content = "\n".join(text_parts)
        return ChatCompletionToolMessageParam(
            role="tool",
            content=text_content,
            tool_call_id=self.tool_use_id,
        )


class TextBlockParamModel(BaseModel):
    type: Literal["text"] = "text"
    text: str
    thinking_signature: bytes | None = Field(default=None)

    def openai_type(self) -> ChatCompletionContentPartTextParam:
        return ChatCompletionContentPartTextParam(type="text", text=self.text)


class ImageBlockParamModel(BaseModel):
    class URLImageSourceParamModel(BaseModel):
        type: Literal["url"] = "url"
        url: str

    type: Literal["image"] = "image"
    source: URLImageSourceParamModel

    def openai_type(self) -> ChatCompletionContentPartImageParam:
        return ChatCompletionContentPartImageParam(
            type="image_url",
            image_url={"url": self.source.url, "detail": "high"},
        )


def _tool_input_to_gemini_args(inp: object) -> dict[str, object]:
    if isinstance(inp, dict):
        return {str(k): v for k, v in inp.items()}
    if isinstance(inp, str):
        try:
            parsed = json.loads(inp)
            if isinstance(parsed, dict):
                return {str(k): v for k, v in parsed.items()}
        except json.JSONDecodeError:
            return {}
    return {}


class ToolUseBlockParamModel(BaseModel):
    id: str
    input: object
    name: str
    type: Literal["tool_use"] = "tool_use"
    thinking_signature: bytes | None = Field(default=None)

    def openai_type(self) -> ChatCompletionMessageToolCallParam:
        args = self.input if isinstance(self.input, str) else json.dumps(self.input)
        return ChatCompletionMessageToolCallParam(
            id=self.id,
            function=Function(name=self.name, arguments=args),
            type="function",
        )


class ThinkingBlockParamModel(BaseModel):
    type: Literal["thinking"] = "thinking"
    thinking: str
    signature: str


MessageContentBlock = (
    TextBlockParamModel
    | ImageBlockParamModel
    | ToolUseBlockParamModel
    | ToolResultBlockParamModel
    | ThinkingBlockParamModel
)


class MessageParamModel(BaseModel):
    role: Literal["user", "assistant"]
    content: list[MessageContentBlock]

    def to_openai_message(self) -> list[ChatCompletionMessageParam]:
        openai_messages: list[ChatCompletionMessageParam] = []
        if self.role == "user":
            user_parts: list[ChatCompletionContentPartParam] = []
            tool_results: list[ChatCompletionToolMessageParam] = []
            for block in self.content:
                if isinstance(block, TextBlockParamModel | ImageBlockParamModel):
                    user_parts.append(block.openai_type())
                elif isinstance(block, ToolResultBlockParamModel):
                    tool_results.append(block.openai_type())
            if tool_results:
                openai_messages.extend(tool_results)
            if user_parts:
                openai_messages.append(
                    ChatCompletionUserMessageParam(role="user", content=user_parts)
                )
        elif self.role == "assistant":
            assistant_text_parts: list[ChatCompletionContentPartTextParam] = []
            tool_calls: list[ChatCompletionMessageToolCallParam] = []
            for block in self.content:
                if isinstance(block, TextBlockParamModel):
                    assistant_text_parts.append(
                        ChatCompletionContentPartTextParam(type="text", text=block.text)
                    )
                elif isinstance(block, ThinkingBlockParamModel):
                    assistant_text_parts.append(
                        ChatCompletionContentPartTextParam(type="text", text=block.thinking)
                    )
                elif isinstance(block, ToolUseBlockParamModel):
                    tool_calls.append(block.openai_type())
            if assistant_text_parts or tool_calls:
                content = _assistant_text_parts_to_openai_content(assistant_text_parts)
                if tool_calls:
                    openai_messages.append(
                        ChatCompletionAssistantMessageParam(
                            role="assistant",
                            content=content,
                            tool_calls=tool_calls,
                        )
                    )
                else:
                    openai_messages.append(
                        ChatCompletionAssistantMessageParam(
                            role="assistant",
                            content=content,
                        )
                    )
        return openai_messages

    def to_gemini_message(self) -> gemini_types.Content:
        parts: list[gemini_types.Part] = []
        for block in self.content:
            if isinstance(block, TextBlockParamModel):
                p = gemini_types.Part(text=block.text)
                if block.thinking_signature is not None:
                    p.thought_signature = block.thinking_signature
                parts.append(p)
            elif isinstance(block, ToolUseBlockParamModel):
                args_dict = _tool_input_to_gemini_args(block.input)
                fc = gemini_types.FunctionCall(
                    id=block.id,
                    name=block.name,
                    args=args_dict,
                )
                fc_part = gemini_types.Part(function_call=fc)
                if block.thinking_signature is not None:
                    fc_part.thought_signature = block.thinking_signature
                parts.append(fc_part)
            elif isinstance(block, ToolResultBlockParamModel):
                text_payload: str
                if isinstance(block.content, str):
                    text_payload = block.content
                else:
                    text_payload = "\n".join(
                        b.text if isinstance(b, ToolResultTextBlock) else "" for b in block.content
                    )
                parts.append(
                    gemini_types.Part(
                        function_response=gemini_types.FunctionResponse(
                            id=block.tool_use_id,
                            name=block.tool_name,
                            response={"result": text_payload}
                            if not block.is_error
                            else {"error": text_payload},
                        )
                    )
                )
            elif isinstance(block, ThinkingBlockParamModel):
                parts.append(gemini_types.Part(thought=True, text=block.thinking))
            elif isinstance(block, ImageBlockParamModel):
                parts.append(
                    gemini_types.Part(
                        text=f"[Image URL for model: {block.source.url}]",
                    )
                )
        if not parts:
            parts = [gemini_types.Part(text="")]
        if self.role == "user":
            return gemini_types.UserContent(parts=parts)
        return gemini_types.ModelContent(parts=parts)


def flatten_messages_to_openai(
    messages: list[MessageParamModel],
) -> list[ChatCompletionMessageParam]:
    out: list[ChatCompletionMessageParam] = []
    for m in messages:
        out.extend(m.to_openai_message())
    return out


def messages_from_llm_prompt_items(items: Sequence[LLMPromptItem]) -> list[MessageParamModel]:
    """Map stringy LLMPromptItem rows to canonical MessageParamModel (single text block each)."""
    return [
        MessageParamModel(
            role=item.role,
            content=[TextBlockParamModel(type="text", text=item.content)],
        )
        for item in items
    ]
