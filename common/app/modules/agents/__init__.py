"""Agent runtime: tool loop, protocol, subagents."""

from common.app.modules.agents.agent import Agent, AgentToolInput
from common.app.modules.agents.builtin_prompts import (
    builtin_system_prompt_fragment,
    builtin_tool_names,
)
from common.app.modules.agents.execution_context import get_acting_agent_name
from common.app.modules.agents.history import MessageHistory
from common.app.modules.agents.protocol import (
    AgentQuestion,
    CopilotProtocol,
    CopilotStreamSinkBridge,
    NullCopilotProtocol,
)
from common.app.modules.agents.spawnable import (
    SPAWN_REGISTRY,
    clear_spawns_for_session,
    get_or_create_spawn,
    make_agent_spawnable,
)
from common.app.modules.agents.subagent_config import SubagentConfig, normalize_subagents
from common.app.modules.agents.subagent_runs import (
    clear_subagent_runs_for_session,
    is_spawn_busy,
    list_runs_for_parent,
    start_delegation,
    wait_all,
    wait_any,
)
from common.app.modules.agents.tools import (
    ImageToolResult,
    Tool,
    ToolResult,
    cancel_tools,
    execute_tools,
)
from common.app.modules.agents.tools.builtin import DELEGATION_BUILTIN_NAMES, builtin_tools


def clear_session(session_id: str) -> int:
    """Release all process-global agent state for a session (spawns + run records).

    Returns the number of entries removed. Call when a session ends to avoid unbounded
    growth of the in-memory registries. Note: these registries are per-process only.
    """
    return clear_spawns_for_session(session_id) + clear_subagent_runs_for_session(session_id)


__all__ = [
    "Agent",
    "AgentQuestion",
    "AgentToolInput",
    "CopilotProtocol",
    "CopilotStreamSinkBridge",
    "DELEGATION_BUILTIN_NAMES",
    "ImageToolResult",
    "MessageHistory",
    "NullCopilotProtocol",
    "SPAWN_REGISTRY",
    "SubagentConfig",
    "Tool",
    "ToolResult",
    "builtin_system_prompt_fragment",
    "builtin_tool_names",
    "builtin_tools",
    "get_acting_agent_name",
    "normalize_subagents",
    "cancel_tools",
    "clear_session",
    "clear_spawns_for_session",
    "clear_subagent_runs_for_session",
    "execute_tools",
    "get_or_create_spawn",
    "is_spawn_busy",
    "list_runs_for_parent",
    "make_agent_spawnable",
    "start_delegation",
    "wait_all",
    "wait_any",
]
