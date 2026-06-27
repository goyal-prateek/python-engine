"""Static system-prompt guidance for built-in agent tools (names only; no Agent import)."""

from __future__ import annotations

# Fixed order for stable prompt composition
ORDERED_BUILTIN_TOOL_NAMES: tuple[str, ...] = (
    "delegate_to_subagent",
    "list_subagent_runs",
    "wait_for_all_subagents",
    "wait_for_any_subagent",
    "run_subagent_sync",
    "ask_user_question",
    "ask_user_permission",
)

BUILTIN_TOOL_GUIDANCE: dict[str, str] = {
    "delegate_to_subagent": (
        "**delegate_to_subagent** — Start a background run for a named sub-agent "
        "(`spawn_name`). The same `spawn_name` reuses that sub-agent's history. "
        "Only one async run per `spawn_name` at a time; use wait/list tools before "
        "delegating again to the same name."
    ),
    "list_subagent_runs": (
        "**list_subagent_runs** — List delegations for this agent (running and finished). "
        "Optional filters: `spawn_names`, `delegation_ids`."
    ),
    "wait_for_all_subagents": (
        "**wait_for_all_subagents** — Block until all matching delegations finish and "
        "return outcomes. Use `include_completed` to wait on already-finished runs too."
    ),
    "wait_for_any_subagent": (
        "**wait_for_any_subagent** — Block until any matching in-flight delegation "
        "completes; other runs keep going."
    ),
    "run_subagent_sync": (
        "**run_subagent_sync** — Run a named sub-agent to completion in this call "
        "(blocks). Do not use if that `spawn_name` already has an async delegation in "
        "progress; wait or use async delegation first."
    ),
    "ask_user_question": (
        "**ask_user_question** — Human-in-the-loop: present questions to the user. "
        "The agent loop pauses until the host resumes with answers via "
        "`run_async(..., human_in_the_loop_tool_id=...)`."
    ),
    "ask_user_permission": (
        "**ask_user_permission** — Human-in-the-loop: ask allow/deny for a sensitive "
        "action. The loop pauses until the host resumes with the tool id and user response."
    ),
}


def builtin_tool_names() -> frozenset[str]:
    return frozenset(BUILTIN_TOOL_GUIDANCE.keys())


def builtin_system_prompt_fragment(*, include: frozenset[str]) -> str:
    """Build a system-preamble fragment for the given builtin tool names (stable order).

    Pass the intersection of tools actually on the agent and the configured allowlist.
    Empty ``include`` returns an empty string.
    """
    if not include:
        return ""
    sections: list[str] = []
    for name in ORDERED_BUILTIN_TOOL_NAMES:
        if name not in include:
            continue
        text = BUILTIN_TOOL_GUIDANCE.get(name)
        if text:
            sections.append(text)
    if not sections:
        return ""
    header = (
        "The following built-in tools are available in this session. "
        "Use them when appropriate; follow each tool's constraints.\n"
    )
    return header + "\n\n".join(sections)


__all__ = [
    "BUILTIN_TOOL_GUIDANCE",
    "ORDERED_BUILTIN_TOOL_NAMES",
    "builtin_system_prompt_fragment",
    "builtin_tool_names",
]
