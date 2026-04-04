"""Named sub-agents with isolated histories sharing the same protocol and backend."""

from __future__ import annotations

import json
import logging
from copy import deepcopy
from typing import Any, Dict, Tuple

from pydantic import BaseModel, Field

from common.app.modules.agents.agent import Agent
from common.app.modules.agents.tools.base import Tool

logger = logging.getLogger(__name__)

SpawnKey = Tuple[str, str, str]

SPAWN_REGISTRY: Dict[SpawnKey, Agent] = {}


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
            "Agent must have a description to be spawnable "
            "(pass description=... to Agent())."
        )
    base = agent

    class SpawnableAgentTool(Tool[SpawnableAgentToolInput]):
        def __init__(self) -> None:
            im = deepcopy(SpawnableAgentToolInput)
            im.__doc__ = base.description or ""
            super().__init__(name=base.name, input_model=im)

        async def execute(self, input: SpawnableAgentToolInput) -> str:
            spawn_name = input.spawn_name
            key: SpawnKey = (base.session_id, base.name, spawn_name)
            if key in SPAWN_REGISTRY:
                spawn = SPAWN_REGISTRY[key]
            else:
                spawn = Agent(
                    name=f"{base.name}#{spawn_name}",
                    system=base.system,
                    session_id=base.session_id,
                    copilot_protocol=base.copilot_protocol,
                    completion=base.completion_backend,
                    tools=list(base.tools),
                    model_config=base.model_config.model_copy(deep=True),
                    description=base.description,
                    stream=False,
                )
                await spawn.initialize()
                SPAWN_REGISTRY[key] = spawn
                logger.info("Created spawn %s for agent %s", spawn_name, base.name)
            blocks = await spawn.run_async(input.agent_input)
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
