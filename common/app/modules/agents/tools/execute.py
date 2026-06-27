"""Execute tool calls from model output with timeouts and validation errors."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping
from typing import Any, TypeVar, cast

from pydantic import BaseModel, ValidationError

from common.app.modules.agents.tools.base import ImageToolResult, Tool, ToolResult
from common.app.modules.llm.messages import (
    ToolResultBlockParamModel,
    ToolUseBlockParamModel,
)

logger = logging.getLogger(__name__)

BaseModelT = TypeVar("BaseModelT", bound=BaseModel)


async def _execute_single_tool(
    call: ToolUseBlockParamModel,
    tool_dict: Mapping[str, Tool[Any]],
) -> ToolResultBlockParamModel:
    response = ToolResultBlockParamModel(
        tool_name=call.name,
        type="tool_result",
        tool_use_id=call.id,
        content="",
        is_error=False,
        human_in_the_loop=False,
    )
    tool: Tool[Any] | None = None
    try:
        tool = tool_dict[call.name]
        tool._current_tool_id = call.id
        tool_input_model = tool.input_model
        raw_input = call.input
        if isinstance(raw_input, tool_input_model):
            validated = raw_input
        else:
            validated = tool_input_model.model_validate(raw_input)

        coro = tool.execute(validated)
        if tool.timeout > 0:
            result = await asyncio.wait_for(coro, timeout=tool.timeout)
        else:
            result = await coro

        if isinstance(result, (ImageToolResult, ToolResult)):
            response.content = result.content
            response.is_error = result.is_error
            if not result.is_error:
                response.human_in_the_loop = result.break_out_of_loop or tool.human_in_the_loop
        else:
            # Plain string results are successful by convention; tools signal errors by
            # returning ToolResult(is_error=True) rather than via a magic "Error:" prefix.
            response.content = str(result)
            response.human_in_the_loop = tool.human_in_the_loop

    except TimeoutError:
        timeout_s = tool.timeout if tool is not None else 0
        response.content = f"Tool execution timed out after {timeout_s} seconds."
        response.is_error = True
        logger.warning("Tool %s timed out (id=%s)", call.name, call.id)
    except KeyError:
        response.content = f"Tool '{call.name}' not found"
        response.is_error = True
    except ValidationError as ve:
        tool = tool_dict.get(call.name)
        if tool is None:
            response.content = f"Tool '{call.name}' not found"
            response.is_error = True
        else:
            input_dict = call.input if isinstance(call.input, dict) else {}
            provided_keys = sorted(str(k) for k in cast(dict[Any, Any], input_dict))
            required_fields = [
                f.alias or name
                for name, f in tool.input_model.model_fields.items()
                if f.is_required()
            ]
            missing = [f for f in required_fields if f not in input_dict]
            if missing:
                response.content = (
                    f"Tool '{call.name}' input validation failed. "
                    f"Provided keys: {provided_keys}. "
                    f"Required: {required_fields}. Missing: {missing}."
                )
            else:
                response.content = f"Tool '{call.name}' validation failed: {ve}"
            response.is_error = True
    except Exception as e:
        response.content = f"Error executing tool: {e!s}"
        response.is_error = True
        logger.exception("Tool %s failed", call.name)

    return response


async def execute_tools(
    tool_calls: list[ToolUseBlockParamModel],
    tool_dict: Mapping[str, Tool[Any]],
    *,
    parallel: bool = True,
) -> list[ToolResultBlockParamModel]:
    if parallel:
        return list(
            await asyncio.gather(*[_execute_single_tool(call, tool_dict) for call in tool_calls])
        )
    out: list[ToolResultBlockParamModel] = []
    for call in tool_calls:
        out.append(await _execute_single_tool(call, tool_dict))
    return out


async def _cancel_single_tool(
    call: ToolUseBlockParamModel,
    tool_dict: Mapping[str, Tool[Any]],
) -> ToolResultBlockParamModel:
    response = ToolResultBlockParamModel(
        tool_name=call.name,
        type="tool_result",
        tool_use_id=call.id,
        content="",
        is_error=True,
    )
    try:
        tool = tool_dict[call.name]
        tool_input_model = tool.input_model
        raw_input = call.input
        if isinstance(raw_input, tool_input_model):
            validated = raw_input
        else:
            validated = tool_input_model.model_validate(raw_input)
        response.content = await tool.cancel(validated)
    except Exception:
        response.content = "cancelled"
    return response


async def cancel_tools(
    tool_calls: list[ToolUseBlockParamModel],
    tool_dict: Mapping[str, Tool[Any]],
) -> list[ToolResultBlockParamModel]:
    return list(
        await asyncio.gather(*[_cancel_single_tool(call, tool_dict) for call in tool_calls])
    )
