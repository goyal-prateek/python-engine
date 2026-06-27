"""Async execution context for attributing CopilotProtocol hooks to an agent name."""

from __future__ import annotations

import contextvars

acting_agent_name: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "acting_agent_name",
    default=None,
)


def get_acting_agent_name() -> str | None:
    """Name of the agent currently running :meth:`Agent.run_async` / ``_agent_loop``.

    When parent and children share one :class:`~common.app.modules.agents.protocol.CopilotProtocol`,
    read this at the **start** of each hook if you defer work to another task; snapshot the
    value into your event payload instead of reading the var only inside deferred callbacks.
    """
    return acting_agent_name.get()


__all__ = ["acting_agent_name", "get_acting_agent_name"]
