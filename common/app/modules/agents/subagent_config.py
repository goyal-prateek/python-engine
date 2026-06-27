"""Per-spawn subagent allowlist and optional overrides."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from common.app.modules.agents.protocol import CopilotProtocol
from common.app.modules.llm.agent_completion import AgentModelConfig


@dataclass
class SubagentConfig:
    """Configuration for one allowed spawn name on a parent agent.

    When used inside a mapping ``{spawn_name: SubagentConfig(...)}``, ``spawn_name``
    is the dict key; leave :attr:`spawn_name` unset on the value.

    When building from a list, set :attr:`spawn_name` on each entry; it is stripped
    when normalized to a dict.
    """

    spawn_name: str | None = None
    system: str | None = None
    tools: list[Any] | None = None
    agent_model_config: AgentModelConfig | None = None
    description: str | None = None
    copilot_protocol: CopilotProtocol | None = None
    stream: bool = False


def normalize_subagents(
    subagents: Mapping[str, SubagentConfig] | Sequence[SubagentConfig] | None,
) -> dict[str, SubagentConfig]:
    """Normalize constructor input to a spawn allowlist dict."""
    if subagents is None:
        return {}
    if isinstance(subagents, Mapping):
        return dict(subagents)
    out: dict[str, SubagentConfig] = {}
    for cfg in subagents:
        if not cfg.spawn_name:
            raise ValueError("SubagentConfig in sequence form must set spawn_name")
        key = cfg.spawn_name
        out[key] = replace(cfg, spawn_name=None)
    return out


__all__ = ["SubagentConfig", "normalize_subagents"]
