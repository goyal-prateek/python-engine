"""Track async subagent delegations (tasks, outcomes) per parent agent.

State here is process-global and keyed by ``session_id``; it is **not** shared across
multiple worker processes. Finished run records are bounded per parent (see
``_MAX_FINISHED_RUNS_PER_PARENT``); call :func:`clear_subagent_runs_for_session` (or
``clear_session`` from ``common.app.modules.agents``) when a session ends to release state.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from common.app.modules.agents.spawnable import SpawnKey, get_or_create_spawn

if TYPE_CHECKING:
    from common.app.modules.agents.agent import Agent

logger = logging.getLogger(__name__)

RunState = Literal["running", "completed", "failed", "cancelled"]

# Cap retained finished (completed/failed/cancelled) run records per parent so a
# long-lived session does not grow the registry without bound.
_MAX_FINISHED_RUNS_PER_PARENT = 50


def _preview_from_blocks(blocks: list[Any], max_len: int = 400) -> str:
    from common.app.modules.llm.messages import TextBlockParamModel, ToolUseBlockParamModel

    parts: list[str] = []
    for b in blocks:
        if isinstance(b, TextBlockParamModel):
            parts.append(b.text.strip())
        elif isinstance(b, ToolUseBlockParamModel):
            parts.append(f"[tool {b.name}]")
    text = " ".join(parts).strip()
    if len(text) > max_len:
        return text[: max_len - 1] + "…"
    return text


@dataclass
class SubagentRunRecord:
    delegation_id: str
    spawn_key: SpawnKey
    task: asyncio.Task[list[Any]]
    state: RunState = "running"
    started_at: float = field(default_factory=time.monotonic)
    finished_at: float | None = None
    error: str | None = None
    result_preview: str | None = None
    result_blocks_json: str | None = None


_registry_lock = asyncio.Lock()
_runs: dict[str, SubagentRunRecord] = {}
_parent_run_ids: dict[tuple[str, str], list[str]] = {}
_running_spawn_keys: set[SpawnKey] = set()


def _parent_key(session_id: str, parent_name: str) -> tuple[str, str]:
    return (session_id, parent_name)


def _prune_finished_for_parent(pk: tuple[str, str]) -> None:
    """Drop oldest finished run records for a parent beyond the retention cap.

    Caller must hold ``_registry_lock``. Running records are always kept.
    """
    ids = _parent_run_ids.get(pk)
    if not ids:
        return
    finished = [did for did in ids if (r := _runs.get(did)) is not None and r.task.done()]
    excess = len(finished) - _MAX_FINISHED_RUNS_PER_PARENT
    if excess <= 0:
        return
    for did in finished[:excess]:
        _runs.pop(did, None)
    keep = [did for did in ids if did in _runs]
    _parent_run_ids[pk] = keep


def _finalize_run_record(rec: SubagentRunRecord, t: asyncio.Task[list[Any]]) -> None:
    rec.finished_at = time.monotonic()
    if t.cancelled():
        rec.state = "cancelled"
        rec.error = "cancelled"
        return
    exc = t.exception()
    if exc is not None:
        rec.state = "failed"
        rec.error = str(exc)
        logger.error("Subagent run %s failed: %s", rec.delegation_id, exc)
        return
    try:
        blocks = t.result()
    except Exception as e:
        rec.state = "failed"
        rec.error = str(e)
        return
    rec.state = "completed"
    rec.result_preview = _preview_from_blocks(blocks)
    rec.result_blocks_json = json.dumps(
        [b.model_dump(exclude_none=True) for b in blocks],
        default=str,
    )


async def start_delegation(parent: Agent, spawn_name: str, agent_input: str) -> str:
    """Start child.run_async in a task; returns delegation_id. Raises ValueError if spawn busy."""
    spawn_key: SpawnKey = (parent.session_id, parent.name, spawn_name)
    async with _registry_lock:
        if spawn_key in _running_spawn_keys:
            raise ValueError(
                f"Subagent spawn {spawn_name!r} already has a run in progress for this parent. "
                "Wait for it to finish before delegating again."
            )
        child = await get_or_create_spawn(parent, spawn_name)
        delegation_id = str(uuid.uuid4())
        _running_spawn_keys.add(spawn_key)

        async def _runner() -> list[Any]:
            try:
                return await child.run_async(agent_input)
            finally:
                async with _registry_lock:
                    _running_spawn_keys.discard(spawn_key)

        task = asyncio.create_task(_runner())
        rec = SubagentRunRecord(delegation_id=delegation_id, spawn_key=spawn_key, task=task)
        _runs[delegation_id] = rec
        pk = _parent_key(parent.session_id, parent.name)
        _parent_run_ids.setdefault(pk, []).append(delegation_id)
        _prune_finished_for_parent(pk)

        def _on_done(t: asyncio.Task[list[Any]]) -> None:
            r = _runs.get(delegation_id)
            if r is None:
                return
            _finalize_run_record(r, t)

        task.add_done_callback(_on_done)
    return delegation_id


def _effective_state(rec: SubagentRunRecord) -> RunState | Literal["running"]:
    if not rec.task.done():
        return "running"
    return rec.state


def _record_public_view(rec: SubagentRunRecord) -> dict[str, Any]:
    session_id, parent_name, spawn_name = rec.spawn_key
    st = _effective_state(rec)
    out: dict[str, Any] = {
        "delegation_id": rec.delegation_id,
        "session_id": session_id,
        "parent_agent_name": parent_name,
        "spawn_name": spawn_name,
        "state": st,
        "started_at_monotonic": rec.started_at,
    }
    if rec.finished_at is not None:
        out["finished_at_monotonic"] = rec.finished_at
    if rec.error:
        out["error"] = rec.error
    if rec.result_preview:
        out["result_preview"] = rec.result_preview
    return out


def list_runs_for_parent(
    session_id: str,
    parent_name: str,
    *,
    spawn_names: set[str] | None = None,
    delegation_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    pk = _parent_key(session_id, parent_name)
    ids = list(_parent_run_ids.get(pk, []))
    rows: list[dict[str, Any]] = []
    for did in ids:
        rec = _runs.get(did)
        if rec is None:
            continue
        if delegation_ids is not None and rec.delegation_id not in delegation_ids:
            continue
        _, _, spawn_name = rec.spawn_key
        if spawn_names is not None and spawn_name not in spawn_names:
            continue
        rows.append(_record_public_view(rec))
    return rows


def _matching_records(
    session_id: str,
    parent_name: str,
    *,
    spawn_names: set[str] | None = None,
    delegation_ids: set[str] | None = None,
    only_running: bool = False,
) -> list[SubagentRunRecord]:
    pk = _parent_key(session_id, parent_name)
    out: list[SubagentRunRecord] = []
    for did in _parent_run_ids.get(pk, []):
        rec = _runs.get(did)
        if rec is None:
            continue
        if delegation_ids is not None and rec.delegation_id not in delegation_ids:
            continue
        _, _, spawn_name = rec.spawn_key
        if spawn_names is not None and spawn_name not in spawn_names:
            continue
        if only_running and rec.task.done():
            continue
        out.append(rec)
    return out


async def wait_all(
    session_id: str,
    parent_name: str,
    *,
    spawn_names: list[str] | None = None,
    delegation_ids: list[str] | None = None,
    only_running: bool = True,
) -> list[dict[str, Any]]:
    spawn_set = set(spawn_names) if spawn_names else None
    did_set = set(delegation_ids) if delegation_ids else None
    async with _registry_lock:
        records = _matching_records(
            session_id,
            parent_name,
            spawn_names=spawn_set,
            delegation_ids=did_set,
            only_running=only_running,
        )
        tasks = [r.task for r in records]
    if not tasks:
        return []
    await asyncio.gather(*tasks, return_exceptions=True)
    return [_wait_one_result_sync(r) for r in records]


async def wait_any(
    session_id: str,
    parent_name: str,
    *,
    spawn_names: list[str] | None = None,
    delegation_ids: list[str] | None = None,
) -> dict[str, Any] | None:
    spawn_set = set(spawn_names) if spawn_names else None
    did_set = set(delegation_ids) if delegation_ids else None
    async with _registry_lock:
        records = _matching_records(
            session_id,
            parent_name,
            spawn_names=spawn_set,
            delegation_ids=did_set,
            only_running=True,
        )
        tasks = [r.task for r in records]
    if not tasks:
        return None
    done, _pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    if not done:
        return None
    t = next(iter(done))
    for r in records:
        if r.task is t:
            await r.task
            return _wait_one_result_sync(r)
    return None


def _wait_one_result_sync(rec: SubagentRunRecord) -> dict[str, Any]:
    base = _record_public_view(rec)
    base["result_blocks_json"] = rec.result_blocks_json
    return base


def is_spawn_busy(session_id: str, parent_name: str, spawn_name: str) -> bool:
    return (session_id, parent_name, spawn_name) in _running_spawn_keys


@asynccontextmanager
async def spawn_busy_guard(
    session_id: str,
    parent_name: str,
    spawn_name: str,
) -> AsyncIterator[None]:
    """Atomically claim a spawn for a synchronous run; release on exit.

    Shares ``_running_spawn_keys`` with :func:`start_delegation` so a sync run and an
    async delegation (or two sync runs) for the same spawn cannot execute concurrently
    against the spawn's shared history. Raises ``ValueError`` if already busy.
    """
    spawn_key: SpawnKey = (session_id, parent_name, spawn_name)
    async with _registry_lock:
        if spawn_key in _running_spawn_keys:
            raise ValueError(
                f"Subagent spawn {spawn_name!r} already has a run in progress for this parent. "
                "Wait for it to finish before running it again."
            )
        _running_spawn_keys.add(spawn_key)
    try:
        yield
    finally:
        async with _registry_lock:
            _running_spawn_keys.discard(spawn_key)


def clear_subagent_runs_for_session(session_id: str) -> int:
    """Remove run records for a session. Does not cancel tasks. Returns removed count."""
    to_del: list[str] = []
    for did, rec in list(_runs.items()):
        sid, _, _ = rec.spawn_key
        if sid == session_id:
            to_del.append(did)
    for did in to_del:
        del _runs[did]
    for k in list(_running_spawn_keys):
        if k[0] == session_id:
            _running_spawn_keys.discard(k)
    pk_to_scrub = [pk for pk in _parent_run_ids if pk[0] == session_id]
    for pk in pk_to_scrub:
        del _parent_run_ids[pk]
    return len(to_del)


__all__ = [
    "clear_subagent_runs_for_session",
    "is_spawn_busy",
    "list_runs_for_parent",
    "spawn_busy_guard",
    "start_delegation",
    "wait_all",
    "wait_any",
]
