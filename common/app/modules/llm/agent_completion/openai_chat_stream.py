"""Accumulate OpenAI-compatible streaming chat completions (OpenRouter) into a message."""

from __future__ import annotations

from typing import Any

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionChunk

from common.app.modules.llm.agent_completion.stream_sink import AgentStreamSink
from common.app.modules.llm.messages import (
    CompletionMessageModel,
    MessageContentBlock,
    MessageParamModel,
    StopReason,
    TextBlockParamModel,
    ThinkingBlockParamModel,
    ToolUseBlockParamModel,
    Usage,
)
from common.app.modules.llm.messages.completion import (
    _openai_usage_nested_int,
    _openai_usage_scalar,
    _parse_openai_tool_arguments,
)


def _reasoning_chunk_to_str(raw: object) -> str:
    if raw is None:
        return ""
    if isinstance(raw, str):
        return raw
    return str(raw)


def _merge_tool_call_delta(
    by_index: dict[int, dict[str, str | None]],
    tc: Any,
) -> None:
    idx = tc.index
    slot = by_index.setdefault(idx, {"id": None, "name": None, "arguments": ""})
    if getattr(tc, "id", None):
        slot["id"] = tc.id
    fn = getattr(tc, "function", None)
    if fn is not None:
        if getattr(fn, "name", None):
            slot["name"] = fn.name
        if getattr(fn, "arguments", None):
            slot["arguments"] = (slot["arguments"] or "") + (fn.arguments or "")


def _finish_to_stop(finish_reason: str | None) -> StopReason | None:
    if finish_reason == "length":
        return "max_tokens"
    if finish_reason in ("tool_calls", "function_call"):
        return "tool_use"
    if finish_reason == "stop":
        return "end_turn"
    if finish_reason == "content_filter":
        return "content_filtered"
    return None


def _usage_from_openai(ou: object | None) -> Usage:
    if ou is None:
        return Usage()
    reasoning_extra = _openai_usage_nested_int(ou, "completion_tokens_details", "reasoning_tokens")
    cached = _openai_usage_nested_int(ou, "prompt_tokens_details", "cached_tokens")
    completion_n = _openai_usage_scalar(ou, "completion_tokens")
    return Usage(
        input_tokens=_openai_usage_scalar(ou, "prompt_tokens"),
        output_tokens=completion_n + reasoning_extra,
        cache_read_input_tokens=cached,
        cache_creation_input_tokens=0,
    )


async def accumulate_openai_chat_stream(
    client: AsyncOpenAI,
    *,
    create_kwargs: dict[str, Any],
    stream_sink: AgentStreamSink,
    fallback_model: str,
) -> CompletionMessageModel:
    await stream_sink.on_stream_start()
    text_parts: list[str] = []
    thinking_parts: list[str] = []
    tool_slots: dict[int, dict[str, str | None]] = {}
    completion_id = ""
    model_id = ""
    finish_reason: str | None = None
    last_usage: object | None = None
    try:
        stream = await client.chat.completions.create(**create_kwargs)
        async for chunk in stream:
            await stream_sink.on_stream_event()
            if not isinstance(chunk, ChatCompletionChunk):
                continue
            if chunk.id:
                completion_id = chunk.id
            if chunk.model:
                model_id = chunk.model
            # OpenAI SDK <1.14-style chunks omit `usage` on the type; OpenRouter may still
            # send it when stream_options.include_usage is honored — use getattr to avoid
            # Pydantic raising on missing fields.
            chunk_usage = getattr(chunk, "usage", None)
            if chunk_usage is not None:
                last_usage = chunk_usage
            for choice in chunk.choices or []:
                if choice.finish_reason is not None:
                    finish_reason = choice.finish_reason
                d = choice.delta
                if d is None:
                    continue
                reasoning_raw = getattr(d, "reasoning", None)
                if reasoning_raw is not None:
                    piece = _reasoning_chunk_to_str(reasoning_raw)
                    if piece:
                        await stream_sink.on_thinking_chunk(piece)
                        thinking_parts.append(piece)
                if d.content:
                    await stream_sink.on_text_chunk(d.content)
                    text_parts.append(d.content)
                if d.tool_calls:
                    for tc in d.tool_calls:
                        _merge_tool_call_delta(tool_slots, tc)
    finally:
        await stream_sink.on_stream_complete()

    thinking_combined = "".join(thinking_parts)
    text_combined = "".join(text_parts)
    content_blocks: list[MessageContentBlock] = []
    if thinking_combined.strip():
        content_blocks.append(
            ThinkingBlockParamModel(type="thinking", thinking=thinking_combined, signature="")
        )
    if text_combined:
        content_blocks.append(TextBlockParamModel(type="text", text=text_combined))

    ordered = sorted(tool_slots.items(), key=lambda x: x[0])
    for _, slot in ordered:
        tid = slot.get("id") or ""
        name = slot.get("name") or ""
        args_raw = slot.get("arguments") or ""
        if not name and not tid and not args_raw.strip():
            continue
        args = _parse_openai_tool_arguments(args_raw) if args_raw.strip() else {}
        content_blocks.append(
            ToolUseBlockParamModel(
                id=tid or f"call_{name}_{len(content_blocks)}",
                input=args,
                name=name or "unknown",
                type="tool_use",
            )
        )

    if not completion_id:
        completion_id = "stream"
    if not model_id:
        model_id = fallback_model

    return CompletionMessageModel(
        id=completion_id,
        type="message",
        content=MessageParamModel(role="assistant", content=content_blocks),
        role="assistant",
        model=model_id,
        usage=_usage_from_openai(last_usage),
        provider="openai",
        stop_reason=_finish_to_stop(finish_reason),
    )
