"""Built-in tools: async subagents, waits, status, and HITL ask/permission."""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field

from common.app.modules.agents.agent import Agent
from common.app.modules.agents.builtin_prompts import builtin_tool_names
from common.app.modules.agents.protocol import AgentQuestion
from common.app.modules.agents.spawnable import get_or_create_spawn
from common.app.modules.agents.subagent_runs import (
    is_spawn_busy,
    list_runs_for_parent,
    start_delegation,
    wait_all,
    wait_any,
)
from common.app.modules.agents.tools.base import Tool, ToolResult

DELEGATION_BUILTIN_NAMES: frozenset[str] = frozenset(
    {
        "delegate_to_subagent",
        "list_subagent_runs",
        "wait_for_all_subagents",
        "wait_for_any_subagent",
        "run_subagent_sync",
    }
)


class DelegateToSubagentInput(BaseModel):
    """Start a named sub-agent run in the background. Same spawn_name shares history.

    Use list_subagent_runs to inspect progress; wait_for_all_subagents or
    wait_for_any_subagent to collect results. Only one run per spawn_name at a time.
    """

    spawn_name: str = Field(..., description="Stable name; reuses the same sub-agent state")
    agent_input: str = Field(..., description="User message / task for the sub-agent")


class ListSubagentRunsInput(BaseModel):
    """List tracked sub-agent delegations for this agent (running and finished)."""

    spawn_names: list[str] | None = Field(
        default=None,
        description="If set, only runs for these spawn names",
    )
    delegation_ids: list[str] | None = Field(
        default=None,
        description="If set, only these delegation ids",
    )


class WaitForAllSubagentsInput(BaseModel):
    """Wait until all matching delegations finish and return their outcomes."""

    spawn_names: list[str] | None = Field(default=None, description="Filter by spawn name")
    delegation_ids: list[str] | None = Field(default=None, description="Filter by delegation id")
    include_completed: bool = Field(
        default=False,
        description="If true, also include already-finished runs in the wait set",
    )


class WaitForAnySubagentInput(BaseModel):
    """Wait until any matching in-flight delegation completes; others keep running."""

    spawn_names: list[str] | None = Field(default=None, description="Filter by spawn name")
    delegation_ids: list[str] | None = Field(default=None, description="Filter by delegation id")


class RunSubagentSyncInput(BaseModel):
    """Run a named sub-agent to completion in this call (blocks until done)."""

    spawn_name: str = Field(..., description="Stable name; reuses the same sub-agent state")
    agent_input: str = Field(..., description="User message / task for the sub-agent")


class AskUserQuestionInput(BaseModel):
    """Ask the human one or more questions; loop pauses until answers are submitted."""

    questions: list[AgentQuestion] = Field(..., description="Questions to present")
    context: str | None = Field(default=None, description="Optional extra context for the host UI")


class AskUserPermissionInput(BaseModel):
    """Ask the human to allow or cancel an action; loop pauses until they respond."""

    title: str = Field(..., description="Short title for the permission UI")
    body: str = Field(..., description="What you are asking permission for")
    context: str | None = Field(default=None, description="Optional extra context for the host UI")


def builtin_tools(agent: Agent) -> list[Tool[Any]]:
    """Return built-in tools bound to ``agent`` (requires ``description`` on the agent).

    Subagent/delegation tools are included only when ``agent.subagents`` is non-empty.
    Respect :attr:`Agent.enabled_builtin_tools` (``None`` = all builtins that are built;
    empty set = no builtins).
    """
    if not agent.description:
        raise ValueError("Agent must have description=... for builtin_tools().")
    base = agent

    class DelegateToSubagentTool(Tool[DelegateToSubagentInput]):
        def __init__(self) -> None:
            super().__init__(name="delegate_to_subagent", input_model=DelegateToSubagentInput)

        async def execute(self, input: DelegateToSubagentInput) -> str | ToolResult:
            try:
                did = await start_delegation(base, input.spawn_name, input.agent_input)
            except ValueError as e:
                return ToolResult(content=str(e), is_error=True)
            return json.dumps(
                {
                    "delegation_id": did,
                    "spawn_name": input.spawn_name,
                    "status": "started",
                }
            )

    class ListSubagentRunsTool(Tool[ListSubagentRunsInput]):
        def __init__(self) -> None:
            super().__init__(name="list_subagent_runs", input_model=ListSubagentRunsInput)

        async def execute(self, input: ListSubagentRunsInput) -> str:
            rows = list_runs_for_parent(
                base.session_id,
                base.name,
                spawn_names=set(input.spawn_names) if input.spawn_names else None,
                delegation_ids=set(input.delegation_ids) if input.delegation_ids else None,
            )
            return json.dumps(rows)

    class WaitForAllSubagentsTool(Tool[WaitForAllSubagentsInput]):
        def __init__(self) -> None:
            super().__init__(name="wait_for_all_subagents", input_model=WaitForAllSubagentsInput)

        async def execute(self, input: WaitForAllSubagentsInput) -> str:
            rows = await wait_all(
                base.session_id,
                base.name,
                spawn_names=input.spawn_names,
                delegation_ids=input.delegation_ids,
                only_running=not input.include_completed,
            )
            return json.dumps(rows)

    class WaitForAnySubagentTool(Tool[WaitForAnySubagentInput]):
        def __init__(self) -> None:
            super().__init__(name="wait_for_any_subagent", input_model=WaitForAnySubagentInput)

        async def execute(self, input: WaitForAnySubagentInput) -> str:
            row = await wait_any(
                base.session_id,
                base.name,
                spawn_names=input.spawn_names,
                delegation_ids=input.delegation_ids,
            )
            if row is None:
                return json.dumps({"detail": "no matching in-flight delegations"})
            return json.dumps(row)

    class RunSubagentSyncTool(Tool[RunSubagentSyncInput]):
        def __init__(self) -> None:
            super().__init__(name="run_subagent_sync", input_model=RunSubagentSyncInput)

        async def execute(self, input: RunSubagentSyncInput) -> str | ToolResult:
            if is_spawn_busy(base.session_id, base.name, input.spawn_name):
                return ToolResult(
                    content=(
                        "Error: this spawn already has an async run in progress. "
                        "Use wait_for_all_subagents / wait_for_any_subagent first."
                    ),
                    is_error=True,
                )
            try:
                spawn = await get_or_create_spawn(base, input.spawn_name)
                blocks = await spawn.run_async(input.agent_input)
            except ValueError as e:
                return ToolResult(content=str(e), is_error=True)
            return json.dumps([b.model_dump(exclude_none=True) for b in blocks])

    class AskUserQuestionTool(Tool[AskUserQuestionInput]):
        def __init__(self) -> None:
            super().__init__(name="ask_user_question", input_model=AskUserQuestionInput)

        async def execute(self, input: AskUserQuestionInput) -> ToolResult:
            tool_id = self._current_tool_id or ""
            await base.copilot_protocol.on_question_request(
                tool_id,
                input.questions,
                input.context,
            )
            payload = {
                "pending_human_input": True,
                "tool": "ask_user_question",
                "questions": [q.model_dump() for q in input.questions],
                "context": input.context,
                "resume_hint": (
                    "Host should call run_async with human_in_the_loop_tool_id=<this tool id> "
                    "and user message containing answers (e.g. JSON keyed by question id)."
                ),
            }
            return ToolResult(content=json.dumps(payload), break_out_of_loop=True)

    class AskUserPermissionTool(Tool[AskUserPermissionInput]):
        def __init__(self) -> None:
            super().__init__(name="ask_user_permission", input_model=AskUserPermissionInput)

        async def execute(self, input: AskUserPermissionInput) -> ToolResult:
            tool_id = self._current_tool_id or ""
            await base.copilot_protocol.on_permission_request(
                tool_id,
                title=input.title,
                body=input.body,
                context=input.context,
            )
            payload = {
                "pending_human_input": True,
                "tool": "ask_user_permission",
                "title": input.title,
                "body": input.body,
                "context": input.context,
                "resume_hint": (
                    "Host should call run_async with human_in_the_loop_tool_id=<this tool id> "
                    "and user message e.g. "
                    '{"allowed": true|false, "message": "optional note from user"}'
                ),
            }
            return ToolResult(content=json.dumps(payload), break_out_of_loop=True)

    delegation_tools: list[Tool[Any]] = [
        DelegateToSubagentTool(),
        ListSubagentRunsTool(),
        WaitForAllSubagentsTool(),
        WaitForAnySubagentTool(),
        RunSubagentSyncTool(),
    ]
    hitl_tools: list[Tool[Any]] = [
        AskUserQuestionTool(),
        AskUserPermissionTool(),
    ]
    tools_out = delegation_tools + hitl_tools if base.subagents else list(hitl_tools)

    known = builtin_tool_names()
    if base.enabled_builtin_tools is not None:
        if len(base.enabled_builtin_tools) == 0:
            return []
        allow = base.enabled_builtin_tools & known
        tools_out = [t for t in tools_out if t.name in allow]
    return tools_out


__all__ = [
    "AskUserPermissionInput",
    "AskUserQuestionInput",
    "DELEGATION_BUILTIN_NAMES",
    "DelegateToSubagentInput",
    "ListSubagentRunsInput",
    "RunSubagentSyncInput",
    "WaitForAllSubagentsInput",
    "WaitForAnySubagentInput",
    "builtin_tools",
]
