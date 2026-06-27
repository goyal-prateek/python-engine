"""Agent loop: LLM + tools until end_turn or stop conditions."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable, Mapping, Sequence
from copy import deepcopy
from typing import Any, cast

from pydantic import BaseModel, Field

from common.app.modules.agents.builtin_prompts import (
    builtin_system_prompt_fragment,
    builtin_tool_names,
)
from common.app.modules.agents.execution_context import acting_agent_name
from common.app.modules.agents.history import MessageHistory
from common.app.modules.agents.protocol import CopilotProtocol, CopilotStreamSinkBridge
from common.app.modules.agents.subagent_config import SubagentConfig, normalize_subagents
from common.app.modules.agents.tools.base import Tool
from common.app.modules.agents.tools.execute import cancel_tools, execute_tools
from common.app.modules.llm.agent_completion import (
    AgentCompletionRequest,
    AgentModelConfig,
    CompletionBackend,
)
from common.app.modules.llm.messages import (
    MessageContentBlock,
    MessageParamModel,
    TextBlockParamModel,
    ToolResultBlockParamModel,
    ToolUseBlockParamModel,
)

logger = logging.getLogger(__name__)

MAX_CONSECUTIVE_IDENTICAL_ERRORS = 3


class AgentToolInput(BaseModel):
    agent_input: str = Field(..., description="The input to the agent")


class Agent:
    MAX_CONSECUTIVE_IDENTICAL_ERRORS = MAX_CONSECUTIVE_IDENTICAL_ERRORS

    def __init__(
        self,
        *,
        name: str,
        system: str,
        session_id: str,
        copilot_protocol: CopilotProtocol,
        completion: CompletionBackend,
        tools: list[Tool[Any]] | None = None,
        model_config: AgentModelConfig | None = None,
        description: str | None = None,
        stream: bool = False,
        on_iteration: Callable[..., Awaitable[None]] | None = None,
        subagents: Mapping[str, SubagentConfig] | Sequence[SubagentConfig] | None = None,
        enabled_builtin_tools: frozenset[str] | set[str] | None = None,
    ) -> None:
        self.name = name
        self.system = system
        self.session_id = session_id
        self.copilot_protocol = copilot_protocol
        self._completion = completion
        self.tools: list[Tool[Any]] = list(tools or [])
        self.model_config = model_config or AgentModelConfig(
            provider="gemini",
            model="gemini-2.0-flash",
        )
        self.description = description
        self.stream = stream
        self._on_iteration = on_iteration
        self.subagents: dict[str, SubagentConfig] = normalize_subagents(subagents)
        if enabled_builtin_tools is not None:
            eb = frozenset(enabled_builtin_tools)
            unknown = eb - builtin_tool_names()
            if unknown:
                raise ValueError(
                    f"Unknown enabled_builtin_tools names: {sorted(unknown)}",
                )
            self.enabled_builtin_tools: frozenset[str] | None = eb
        else:
            self.enabled_builtin_tools = None
        self._turn_last_stop_reason = ""
        self._bootstrapped = False
        self._notification_tasks: set[asyncio.Task[None]] = set()
        self.history = MessageHistory(
            context_window_tokens=self.model_config.context_window_tokens,
            session_id=session_id,
            refresh_sticky_context=copilot_protocol.get_sticky_context,
        )

    @property
    def completion_backend(self) -> CompletionBackend:
        return self._completion

    def _llm_system_with_builtin_context(self) -> str:
        """Prefix ``system`` with guidance for built-in tools present on this agent."""
        known = builtin_tool_names()
        names_in_tools = frozenset(t.name for t in self.tools if t.name in known)
        if not names_in_tools:
            return self.system
        if self.enabled_builtin_tools is not None:
            if len(self.enabled_builtin_tools) == 0:
                return self.system
            want = self.enabled_builtin_tools & known
        else:
            want = known
        include = frozenset(names_in_tools & want)
        fragment = builtin_system_prompt_fragment(include=include)
        if not fragment:
            return self.system
        return f"{fragment}\n\n{self.system}"

    async def initialize(self) -> None:
        await self.history.bootstrap_with_sticky()
        self._bootstrapped = True

    @staticmethod
    def _tool_call_error_fingerprint(
        tool_calls: list[ToolUseBlockParamModel],
    ) -> str | None:
        if not tool_calls:
            return None
        parts: list[str] = []
        for tc in tool_calls:
            keys = sorted(tc.input.keys()) if isinstance(tc.input, dict) else []
            parts.append(f"{tc.name}:{','.join(keys)}")
        return "|".join(parts)

    def _send_notification(self, update: MessageParamModel) -> None:
        # Keep a strong reference so the task is not garbage-collected mid-flight,
        # and surface (rather than swallow) failures from the host's notification hook.
        task = asyncio.create_task(self.copilot_protocol.send_notification(update))
        self._notification_tasks.add(task)
        task.add_done_callback(self._on_notification_done)

    def _on_notification_done(self, task: asyncio.Task[None]) -> None:
        self._notification_tasks.discard(task)
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            logger.warning("send_notification failed: %s", exc)

    async def _agent_loop(
        self,
        user_input: str,
        human_in_the_loop_tool_id: str | None = None,
        extra_content_blocks: list[Any] | None = None,
    ) -> list[TextBlockParamModel | ToolUseBlockParamModel]:
        self._turn_last_stop_reason = ""

        if not self._bootstrapped:
            await self.initialize()

        unhandled = await self.copilot_protocol.get_unhandled_messages()
        bg = await self.copilot_protocol.get_background_context_messages()
        for message in unhandled + bg:
            await self.history.add_message(
                MessageParamModel(role="user", content=[message]),
                usage=None,
            )

        if human_in_the_loop_tool_id:
            await self.history.add_human_in_the_loop_message(human_in_the_loop_tool_id, user_input)
        else:
            blocks: list[Any] = [TextBlockParamModel(type="text", text=user_input)]
            if extra_content_blocks:
                blocks.extend(extra_content_blocks)
            await self.history.add_message(
                MessageParamModel(
                    role="user",
                    content=cast(list[MessageContentBlock], blocks),
                ),
                usage=None,
            )

        tool_dict: dict[str, Tool[Any]] = {t.name: t for t in self.tools}
        last_error_fingerprint: str | None = None
        consecutive_identical_errors = 0

        while True:
            if not await self.copilot_protocol.should_continue_loop():
                self._turn_last_stop_reason = "aborted"
                break

            self.history.truncate()

            request = AgentCompletionRequest.from_parts(
                messages=self.history.format_for_api(),
                system=self._llm_system_with_builtin_context(),
                tools=tuple(self.tools),
                config=self.model_config,
            )
            stream_sink = CopilotStreamSinkBridge(self.copilot_protocol) if self.stream else None
            response = await self._completion.complete(request, stream_sink=stream_sink)

            if self._on_iteration is not None:
                await self._on_iteration(response=response, agent=self)

            self._send_notification(response.content)

            await self.history.add_message(
                MessageParamModel(
                    role="assistant",
                    content=list(response.content.content),
                ),
                usage=response.usage,
            )

            await self.copilot_protocol.on_context_usage(
                self.history.total_tokens,
                self.history.context_window_tokens,
            )

            if response.stop_reason == "max_tokens":
                tool_calls_in_response = [
                    b for b in response.content.content if isinstance(b, ToolUseBlockParamModel)
                ]
                if tool_calls_in_response:
                    truncation_error = (
                        "Tool call was NOT executed — output truncated at max_tokens."
                    )
                    err_results: list[ToolResultBlockParamModel] = []
                    for tc in tool_calls_in_response:
                        err_results.append(
                            ToolResultBlockParamModel(
                                tool_name=tc.name,
                                type="tool_result",
                                tool_use_id=tc.id,
                                content=truncation_error,
                                is_error=True,
                            )
                        )
                        await self.copilot_protocol.on_tool_result(tc.id, truncation_error, True)
                    await self.history.add_message(
                        MessageParamModel(
                            role="user",
                            content=cast(
                                list[MessageContentBlock],
                                err_results
                                + [
                                    TextBlockParamModel(
                                        type="text",
                                        text="[System] Retry with smaller tool outputs.",
                                    )
                                ],
                            ),
                        ),
                        usage=None,
                    )
                    self._turn_last_stop_reason = "max_tokens"
                    continue

            unhandled_messages = await self.copilot_protocol.get_unhandled_messages()
            tool_calls = [
                b for b in response.content.content if isinstance(b, ToolUseBlockParamModel)
            ]

            if tool_calls:
                if unhandled_messages:
                    tool_results = await cancel_tools(tool_calls, tool_dict)
                else:
                    tool_results = await execute_tools(tool_calls, tool_dict)

                all_errored = all(r.is_error for r in tool_results)
                if all_errored:
                    fp = self._tool_call_error_fingerprint(tool_calls)
                    if fp and fp == last_error_fingerprint:
                        consecutive_identical_errors += 1
                    else:
                        consecutive_identical_errors = 1
                        last_error_fingerprint = fp
                    if consecutive_identical_errors >= self.MAX_CONSECUTIVE_IDENTICAL_ERRORS:
                        for tr in tool_results:
                            disp = tr.content if isinstance(tr.content, str) else str(tr.content)
                            await self.copilot_protocol.on_tool_result(tr.tool_use_id, disp, True)
                        await self.history.add_message(
                            MessageParamModel(
                                role="user",
                                content=cast(
                                    list[MessageContentBlock],
                                    list(tool_results)
                                    + [
                                        TextBlockParamModel(
                                            type="text",
                                            text="[System] Circuit breaker: repeated "
                                            "identical tool errors.",
                                        )
                                    ],
                                ),
                            ),
                            usage=None,
                        )
                        self._turn_last_stop_reason = "circuit_breaker"
                        break
                else:
                    consecutive_identical_errors = 0
                    last_error_fingerprint = None

                for tool_result in tool_results:
                    disp = (
                        tool_result.content
                        if isinstance(tool_result.content, str)
                        else str(tool_result.content)
                    )
                    await self.copilot_protocol.on_tool_result(
                        tool_result.tool_use_id,
                        disp,
                        tool_result.is_error,
                    )

                self._send_notification(MessageParamModel(role="user", content=list(tool_results)))

                new_unhandled = await self.copilot_protocol.get_unhandled_messages()
                new_bg = await self.copilot_protocol.get_background_context_messages()
                combined = list(tool_results) + list(new_unhandled) + list(new_bg)
                await self.history.add_message(
                    MessageParamModel(
                        role="user",
                        content=cast(list[MessageContentBlock], combined),
                    ),
                    usage=None,
                )

                hitl = any(b.human_in_the_loop for b in tool_results)
                if hitl or not await self.copilot_protocol.should_continue_loop():
                    self._turn_last_stop_reason = "human_in_the_loop" if hitl else "aborted"
                    return [
                        b
                        for b in response.content.content
                        if isinstance(b, (TextBlockParamModel, ToolUseBlockParamModel))
                    ]
            else:
                unhandled_messages = await self.copilot_protocol.get_unhandled_messages()
                bg_ctx = await self.copilot_protocol.get_background_context_messages()
                extra_msgs = list(unhandled_messages) + list(bg_ctx)
                if extra_msgs:
                    await self.history.add_message(
                        MessageParamModel(
                            role="user",
                            content=cast(list[MessageContentBlock], extra_msgs),
                        ),
                        usage=None,
                    )
                    continue
                self._turn_last_stop_reason = "end_turn"
                return [
                    b
                    for b in response.content.content
                    if isinstance(b, (TextBlockParamModel, ToolUseBlockParamModel))
                ]

        return []

    async def run_async(
        self,
        user_input: str,
        human_in_the_loop_tool_id: str | None = None,
        extra_content_blocks: list[Any] | None = None,
    ) -> list[TextBlockParamModel | ToolUseBlockParamModel]:
        completed_normally = False
        token = acting_agent_name.set(self.name)
        try:
            await self.copilot_protocol.mark_loop_as_in_progress()
            result = await self._agent_loop(
                user_input,
                human_in_the_loop_tool_id,
                extra_content_blocks,
            )
            completed_normally = True
            return result
        finally:
            acting_agent_name.reset(token)
            if completed_normally:
                await self.copilot_protocol.mark_loop_as_completed()
            else:
                await self.copilot_protocol.mark_loop_as_interrupted()

    async def get_agent_as_tool(self) -> Tool[AgentToolInput]:
        if not self.description:
            raise ValueError(
                "Agent must have a description to be used as a tool "
                "(pass description=... to Agent())."
            )
        run_async = self.run_async
        agent_name = self.name
        desc = self.description

        class AgentTool(Tool[AgentToolInput]):
            def __init__(self) -> None:
                input_model = deepcopy(AgentToolInput)
                input_model.__doc__ = desc
                super().__init__(name=agent_name, input_model=input_model, timeout=0)

            async def execute(self, input: AgentToolInput) -> str:
                blocks = await run_async(input.agent_input)
                payload = [
                    b.model_dump(
                        exclude_none=True,
                        exclude_unset=True,
                        exclude_defaults=True,
                    )
                    for b in blocks
                ]
                return json.dumps(payload)

        return AgentTool()
