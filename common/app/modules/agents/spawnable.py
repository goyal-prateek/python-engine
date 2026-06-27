"""Named sub-agents with isolated histories sharing the same protocol and backend.

``SPAWN_REGISTRY`` is process-global and keyed by ``session_id``; it is **not** shared
across worker processes. Call :func:`clear_spawns_for_session` (or ``clear_session`` from
``common.app.modules.agents``) when a session ends to release spawned agents.
"""

from __future__ import annotations

import json
import logging
from copy import deepcopy
from typing import Any

from pydantic import BaseModel, Field

from common.app.modules.agents.agent import Agent
from common.app.modules.agents.builtin_prompts import builtin_tool_names
from common.app.modules.agents.subagent_config import SubagentConfig
from common.app.modules.agents.tools.base import Tool, ToolResult

logger = logging.getLogger(__name__)

SpawnKey = tuple[str, str, str]

SPAWN_REGISTRY: dict[SpawnKey, Agent] = {}


async def get_or_create_spawn(parent: Agent, spawn_name: str) -> Agent:
    """Return named sub-agent for this parent session, creating and bootstrapping if needed."""
    key: SpawnKey = (parent.session_id, parent.name, spawn_name)
    if key in SPAWN_REGISTRY:
        return SPAWN_REGISTRY[key]
    if not parent.subagents:
        raise ValueError(
            "This agent has no subagents configured; spawning is disabled.",
        )
    if spawn_name not in parent.subagents:
        raise ValueError(
            f"Unknown spawn_name {spawn_name!r}; not in the configured subagent allowlist.",
        )
    cfg: SubagentConfig = parent.subagents[spawn_name]
    system = cfg.system if cfg.system is not None else parent.system
    if cfg.tools is not None:
        tools = list(cfg.tools)
    else:
        # Inherited parent tools include built-in delegation/HITL tools that close over the
        # parent (session_id/name). Strip them so the child does not act as its parent;
        # the child can be given its own built-ins explicitly via cfg.tools.
        known_builtins = builtin_tool_names()
        tools = [t for t in parent.tools if t.name not in known_builtins]
    model_cfg = (
        cfg.agent_model_config
        if cfg.agent_model_config is not None
        else parent.model_config.model_copy(deep=True)
    )
    description = cfg.description if cfg.description is not None else parent.description
    proto = cfg.copilot_protocol if cfg.copilot_protocol is not None else parent.copilot_protocol
    stream = cfg.stream
    spawn = Agent(
        name=f"{parent.name}#{spawn_name}",
        system=system,
        session_id=parent.session_id,
        copilot_protocol=proto,
        completion=parent.completion_backend,
        tools=tools,
        model_config=model_cfg,
        description=description,
        stream=stream,
        subagents={},
        enabled_builtin_tools=parent.enabled_builtin_tools,
    )
    await spawn.initialize()
    SPAWN_REGISTRY[key] = spawn
    logger.info("Created spawn %s for agent %s", spawn_name, parent.name)
    return spawn


class SpawnableAgentToolInput(BaseModel):
    """Run a named sub-agent instance (creates or reuses history per spawn_name)."""

    agent_input: str = Field(..., description="The input to the agent")
    spawn_name: str = Field(
        ...,
        description="Name of the spawn; same name reuses conversation state.",
    )


class SpawnInfoInput(BaseModel):
    """List spawns for this parent agent in the current session."""

    pass


def clear_spawns_for_session(session_id: str) -> int:
    """Remove all registry entries for a session (e.g. tests). Returns count removed."""
    keys = [k for k in SPAWN_REGISTRY if k[0] == session_id]
    for k in keys:
        del SPAWN_REGISTRY[k]
    return len(keys)


async def make_agent_spawnable(agent: Agent) -> list[Tool[Any]]:
    if not agent.description:
        raise ValueError(
            "Agent must have a description to be spawnable (pass description=... to Agent())."
        )
    if not agent.subagents:
        return []
    base = agent

    class SpawnableAgentTool(Tool[SpawnableAgentToolInput]):
        def __init__(self) -> None:
            im = deepcopy(SpawnableAgentToolInput)
            im.__doc__ = base.description or ""
            super().__init__(name=base.name, input_model=im)

        async def execute(self, input: SpawnableAgentToolInput) -> str | ToolResult:
            from common.app.modules.agents.subagent_runs import spawn_busy_guard

            try:
                async with spawn_busy_guard(base.session_id, base.name, input.spawn_name):
                    spawn = await get_or_create_spawn(base, input.spawn_name)
                    blocks = await spawn.run_async(input.agent_input)
            except ValueError as e:
                return ToolResult(content=str(e), is_error=True)
            return json.dumps([b.model_dump(exclude_none=True) for b in blocks])

    class SpawnInfoTool(Tool[SpawnInfoInput]):
        def __init__(self) -> None:
            super().__init__(
                name=f"get_{base.name}_spawn_info",
                input_model=SpawnInfoInput,
            )

        async def execute(self, input: SpawnInfoInput) -> str:
            rows = [
                {
                    "spawn_name": k[2],
                    "messages": len(SPAWN_REGISTRY[k].history.messages),
                }
                for k in SPAWN_REGISTRY
                if k[0] == base.session_id and k[1] == base.name
            ]
            return json.dumps(
                {
                    "agent_name": base.name,
                    "session_id": base.session_id,
                    "spawns": rows,
                }
            )

    return [SpawnableAgentTool(), SpawnInfoTool()]
