"""Assistant completion envelope and provider-specific parsers."""

from __future__ import annotations

import json
import uuid
from typing import Literal

from google.genai import types as gemini_types
from openai.types.chat import ChatCompletion
from pydantic import BaseModel

from common.app.modules.llm.messages.blocks import (
    MessageParamModel,
    TextBlockParamModel,
    ThinkingBlockParamModel,
    ToolResultBlockParamModel,
    ToolUseBlockParamModel,
)


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0


StopReason = Literal[
    "end_turn",
    "max_tokens",
    "tool_use",
    "content_filtered",
    "stop_sequence",
    "pause_turn",
    "malformed_tool_call",
    "recitation",
    "language",
    "no_image",
    "other",
]


def _parse_openai_tool_arguments(raw: str) -> object:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"_raw_arguments": raw, "_parse_error": "invalid_json"}


def _openai_usage_scalar(usage: object, key: str) -> int:
    """Read prompt_tokens / completion_tokens from SDK object or raw dict (e.g. OpenRouter)."""
    if usage is None:
        return 0
    raw = usage.get(key) if isinstance(usage, dict) else getattr(usage, key, None)
    if raw is None:
        return 0
    try:
        return int(raw)
    except (TypeError, ValueError):
        return 0


def _openai_usage_nested_int(usage: object, parent_key: str, child_key: str) -> int:
    """Read nested usage fields; parent may be a typed object or dict."""
    if usage is None:
        return 0
    parent = usage.get(parent_key) if isinstance(usage, dict) else getattr(usage, parent_key, None)
    if parent is None:
        return 0
    child = parent.get(child_key) if isinstance(parent, dict) else getattr(parent, child_key, None)
    if child is None:
        return 0
    try:
        return int(child)
    except (TypeError, ValueError):
        return 0


class CompletionMessageModel(BaseModel):
    id: str
    type: Literal["message"] = "message"
    content: MessageParamModel
    role: Literal["assistant"] = "assistant"
    model: str
    usage: Usage
    provider: Literal["openai", "anthropic", "gemini"]
    stop_reason: StopReason | None = None

    @property
    def text_content(self) -> str:
        parts: list[str] = []
        for block in self.content.content:
            if isinstance(block, TextBlockParamModel):
                parts.append(block.text)
        return "".join(parts)

    @staticmethod
    def from_openai_chat_completion(completion: ChatCompletion) -> CompletionMessageModel:
        completion_content = MessageParamModel(role="assistant", content=[])
        choice = completion.choices[0]
        msg = choice.message
        reasoning_raw = getattr(msg, "reasoning", None)
        if reasoning_raw is not None:
            rs = reasoning_raw if isinstance(reasoning_raw, str) else str(reasoning_raw)
            if rs.strip():
                completion_content.content.append(
                    ThinkingBlockParamModel(
                        type="thinking",
                        thinking=rs,
                        signature="",
                    )
                )
        if msg.content is not None:
            completion_content.content.append(TextBlockParamModel(type="text", text=msg.content))
        if msg.tool_calls is not None:
            for tool_call in msg.tool_calls:
                args = _parse_openai_tool_arguments(tool_call.function.arguments)
                completion_content.content.append(
                    ToolUseBlockParamModel(
                        id=tool_call.id,
                        input=args,
                        name=tool_call.function.name,
                        type="tool_use",
                    )
                )
        ou = completion.usage
        if ou is None:
            usage = Usage()
        else:
            reasoning_extra = _openai_usage_nested_int(
                ou, "completion_tokens_details", "reasoning_tokens"
            )
            cached = _openai_usage_nested_int(ou, "prompt_tokens_details", "cached_tokens")
            completion_n = _openai_usage_scalar(ou, "completion_tokens")
            usage = Usage(
                input_tokens=_openai_usage_scalar(ou, "prompt_tokens"),
                output_tokens=completion_n + reasoning_extra,
                cache_read_input_tokens=cached,
                cache_creation_input_tokens=0,
            )

        finish_reason = choice.finish_reason
        stop_reason: StopReason | None = None
        if finish_reason == "length":
            stop_reason = "max_tokens"
        elif finish_reason in ("tool_calls", "function_call"):
            stop_reason = "tool_use"
        elif finish_reason == "stop":
            stop_reason = "end_turn"
        elif finish_reason == "content_filter":
            stop_reason = "content_filtered"

        return CompletionMessageModel(
            id=completion.id,
            type="message",
            content=completion_content,
            role="assistant",
            model=completion.model,
            usage=usage,
            provider="openai",
            stop_reason=stop_reason,
        )

    @staticmethod
    def from_gemini_message(
        message: gemini_types.GenerateContentResponse,
    ) -> CompletionMessageModel:
        message_param_model = MessageParamModel(role="assistant", content=[])

        if message.candidates and len(message.candidates) > 0:
            candidate = message.candidates[0]
            if candidate.content and candidate.content.parts:
                for part in candidate.content.parts:
                    if part.text is not None:
                        if part.thought:
                            sig = ""
                            if part.thought_signature:
                                try:
                                    sig = part.thought_signature.decode("utf-8")
                                except Exception:
                                    sig = ""
                            message_param_model.content.append(
                                ThinkingBlockParamModel(
                                    type="thinking",
                                    thinking=part.text,
                                    signature=sig,
                                )
                            )
                        else:
                            message_param_model.content.append(
                                TextBlockParamModel(
                                    type="text",
                                    text=part.text,
                                    thinking_signature=part.thought_signature,
                                )
                            )
                    if part.function_call is not None:
                        fc = part.function_call
                        raw_args = fc.args if fc.args is not None else {}
                        message_param_model.content.append(
                            ToolUseBlockParamModel(
                                id=fc.id or f"func_{fc.name or 'unknown'}_{uuid.uuid4().hex[:8]}",
                                input=dict(raw_args) if raw_args else {},
                                name=fc.name or "",
                                type="tool_use",
                                thinking_signature=part.thought_signature,
                            )
                        )
                    if part.function_response is not None:
                        fr = part.function_response
                        payload = json.dumps(fr.response) if fr.response else ""
                        message_param_model.content.append(
                            ToolResultBlockParamModel(
                                tool_name=fr.name or "",
                                tool_use_id=fr.id or f"resp_{fr.name}",
                                type="tool_result",
                                content=payload,
                                is_error=False,
                            )
                        )

        usage = Usage()
        if message.usage_metadata:
            um = message.usage_metadata
            prompt_count = um.prompt_token_count or 0
            cached = um.cached_content_token_count or 0
            usage = Usage(
                input_tokens=prompt_count - cached,
                output_tokens=um.candidates_token_count or 0,
                cache_read_input_tokens=cached,
                cache_creation_input_tokens=0,
            )

        _CONTENT_FILTERED = {
            "SAFETY",
            "PROHIBITED_CONTENT",
            "BLOCKLIST",
            "SPII",
            "IMAGE_SAFETY",
            "IMAGE_PROHIBITED_CONTENT",
        }
        _MALFORMED_TOOL = {"MALFORMED_FUNCTION_CALL", "UNEXPECTED_TOOL_CALL"}

        gemini_stop: StopReason | None = None
        if message.candidates and message.candidates[0].finish_reason:
            finish_r = message.candidates[0].finish_reason
            name = finish_r.name if hasattr(finish_r, "name") else str(finish_r)
            if name == "MAX_TOKENS":
                gemini_stop = "max_tokens"
            elif name == "STOP":
                gemini_stop = "end_turn"
            elif name in _CONTENT_FILTERED:
                gemini_stop = "content_filtered"
            elif name in _MALFORMED_TOOL:
                gemini_stop = "malformed_tool_call"
            elif name == "NO_IMAGE":
                gemini_stop = "no_image"
            else:
                gemini_stop = "other"

        model_ver = message.model_version or "gemini"
        return CompletionMessageModel(
            id=message.response_id or "gemini_response",
            type="message",
            content=message_param_model,
            role="assistant",
            model=model_ver,
            usage=usage,
            provider="gemini",
            stop_reason=gemini_stop,
        )
