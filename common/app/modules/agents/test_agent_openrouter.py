"""Minimal agent for manual OpenRouter + tool-calling checks.

Run from repo root (with OPENROUTER_API_KEY set):

    python -m common.app.modules.agents.test_agent_openrouter

Logs each phase: initial user message → streamed reasoning/text (CopilotProtocol) →
assembled assistant turn → tool results → next turn(s).

This harness always uses streaming completions (same code path as a streaming FE).
With `NullCopilotProtocol`, `on_*_chunk` hooks are no-ops, so you only see assembled
turns if you add logging elsewhere. OpenRouter `reasoning.max_tokens` feeds thinking
chunks when the model supports them.
"""

from __future__ import annotations

import asyncio
import json
import random
import sys
import textwrap
from typing import Any

from pydantic import BaseModel, Field

from common.app.modules.agents import Agent, NullCopilotProtocol, Tool
from common.app.modules.agents.protocol import CopilotProtocol
from common.app.modules.llm.agent_completion import AgentModelConfig, CompletionRouter
from common.app.modules.llm.clients import SharedLLMClients, create_shared_llm_clients
from common.app.modules.llm.messages import (
    CompletionMessageModel,
    TextBlockParamModel,
    ThinkingBlockParamModel,
    ToolUseBlockParamModel,
)

OPENROUTER_TEST_MODEL = "google/gemini-3.1-flash-lite-preview"

TEST_AGENT_SYSTEM = """You are a concise test assistant for integration checks.
When the user asks for math, dice rolls, or structured notes, use the appropriate tools.
For multi-step requests, you may call one tool per turn or batch tools in one turn;
either is fine. Keep final replies short unless the user asks for detail."""

DEMO_USER_MESSAGE = textwrap.dedent("""\
    Roll a d20, then add 41 and 1, then save a note titled 'ping' with body 'ok'.
    When done with tools, reply with one short sentence summarizing what you did.\
""")


class AddInput(BaseModel):
    """Add two numbers."""

    a: float = Field(..., description="First operand")
    b: float = Field(..., description="Second operand")


class AddTool(Tool[AddInput]):
    def __init__(self) -> None:
        super().__init__(name="add_numbers", input_model=AddInput)

    async def execute(self, inp: AddInput) -> str:
        return str(inp.a + inp.b)


class RollInput(BaseModel):
    """Roll an N-sided die (integer sides >= 2)."""

    sides: int = Field(
        ...,
        ge=2,
        le=1000,
        description="Number of sides on the die",
    )


class RollDiceTool(Tool[RollInput]):
    def __init__(self) -> None:
        super().__init__(name="roll_dice", input_model=RollInput)

    async def execute(self, inp: RollInput) -> str:
        return str(random.randint(1, inp.sides))


class NoteInput(BaseModel):
    """Store a short text note (echo back as confirmation)."""

    title: str = Field(..., description="Short title for the note")
    body: str = Field(..., description="Note content")


class SaveNoteTool(Tool[NoteInput]):
    def __init__(self) -> None:
        super().__init__(name="save_note", input_model=NoteInput)

    async def execute(self, inp: NoteInput) -> str:
        return json.dumps({"saved": True, "title": inp.title, "body": inp.body})


def _banner(title: str, char: str = "=") -> None:
    line = char * 72
    print(f"\n{line}\n{title}\n{line}")


def _format_assistant_blocks(response: CompletionMessageModel) -> None:
    blocks = response.content.content
    if not blocks:
        print("  (empty assistant content)")
        return
    for i, b in enumerate(blocks, start=1):
        if isinstance(b, TextBlockParamModel):
            preview = b.text.strip().replace("\n", " ")
            if len(preview) > 200:
                preview = preview[:200] + "…"
            print(f"  [{i}] text: {preview}")
        elif isinstance(b, ToolUseBlockParamModel):
            print(f"  [{i}] tool_use: name={b.name!r} id={b.id!r}")
            print(f"       input: {b.input}")
        elif isinstance(b, ThinkingBlockParamModel):
            t = (b.thinking or "").strip().replace("\n", " ")
            if len(t) > 120:
                t = t[:120] + "…"
            print(f"  [{i}] thinking: {t}")
        else:
            print(f"  [{i}] {type(b).__name__}: {b}")


def _format_final_blocks(blocks: list[Any]) -> str:
    lines: list[str] = []
    for b in blocks:
        if isinstance(b, TextBlockParamModel):
            lines.append(b.text)
        elif isinstance(b, ThinkingBlockParamModel):
            t = (b.thinking or "").strip()
            if len(t) > 300:
                t = t[:300] + "…"
            lines.append(f"[thinking] {t}")
        elif isinstance(b, ToolUseBlockParamModel):
            lines.append(f"[tool_use {b.name} id={b.id} input={b.input}]")
        else:
            lines.append(str(b))
    return "\n".join(lines) if lines else "(no text/tool blocks)"


class DemoTranscriptLog:
    """Shared state: LLM round logging + tool-result banner + per-tool lines."""

    def __init__(self) -> None:
        self.llm_round = 0
        self._tool_results_header_next = False

    def take_tool_results_section_header(self) -> bool:
        """Return True once when starting a batch of tool results (for a section banner)."""
        if not self._tool_results_header_next:
            return False
        self._tool_results_header_next = False
        return True

    async def on_iteration(
        self, *, response: CompletionMessageModel, agent: Agent
    ) -> None:
        self.llm_round += 1
        n = self.llm_round
        _banner(f"TURN {n} — ASSISTANT (model response)")
        u = response.usage
        print(
            f"  usage: input_tokens={u.input_tokens}  "
            f"output_tokens={u.output_tokens}  "
            f"cache_read_input_tokens={u.cache_read_input_tokens}"
        )
        print(f"  stop_reason: {response.stop_reason!r}")
        print("  content blocks:")
        _format_assistant_blocks(response)
        has_tools = any(
            isinstance(b, ToolUseBlockParamModel) for b in response.content.content
        )
        self._tool_results_header_next = has_tools
        if has_tools:
            print("\n  (Agent runs tools next; see TOOL RESULTS section.)\n")


class LoggingTestCopilotProtocol(NullCopilotProtocol):
    """Logs tool results, streaming tokens (thinking vs answer), and stream lifecycle."""

    def __init__(self, transcript: DemoTranscriptLog) -> None:
        self._transcript = transcript
        self._stream_phase: str | None = None

    async def on_stream_start(self) -> None:
        self._stream_phase = None
        print("\n  ── CopilotProtocol: stream start (same hooks FE would use) ──")

    async def on_stream_complete(self) -> None:
        if self._stream_phase is not None:
            sys.stdout.write("\n")
            sys.stdout.flush()
        self._stream_phase = None
        print("  ── CopilotProtocol: stream complete ──\n")

    async def on_stream_event(self) -> None:
        """One chunk envelope from the provider (often many per completion)."""
        pass

    async def on_text_chunk(self, chunk: str) -> None:
        if self._stream_phase != "text":
            sys.stdout.write("\n  [stream · assistant text] ")
            self._stream_phase = "text"
        sys.stdout.write(chunk)
        sys.stdout.flush()

    async def on_thinking_chunk(self, chunk: str) -> None:
        if self._stream_phase != "think":
            sys.stdout.write("\n  [stream · reasoning / thinking] ")
            self._stream_phase = "think"
        sys.stdout.write(chunk)
        sys.stdout.flush()

    async def on_tool_result(
        self, tool_id: str, result: str, is_error: bool = False
    ) -> None:
        if self._transcript.take_tool_results_section_header():
            _banner("TOOL RESULTS — environment → model (next user message in history)")
        status = "error" if is_error else "ok"
        print(f"  • tool_use_id={tool_id!r} ({status})")
        print(textwrap.indent(result, "    "))


def create_test_openrouter_agent(
    shared_clients: SharedLLMClients,
    *,
    session_id: str = "test-openrouter-session",
    copilot_protocol: CopilotProtocol | None = None,
    on_iteration: Any | None = None,
) -> Agent:
    return Agent(
        name="test_openrouter",
        system=TEST_AGENT_SYSTEM,
        session_id=session_id,
        copilot_protocol=copilot_protocol or NullCopilotProtocol(),
        completion=CompletionRouter(shared_clients),
        tools=[AddTool(), RollDiceTool(), SaveNoteTool()],
        model_config=AgentModelConfig(
            provider="openai",
            model=OPENROUTER_TEST_MODEL,
            openrouter_extra_body={
                "reasoning": {"max_tokens": 8192},
            },
        ),
        description="OpenRouter test agent with add_numbers, roll_dice, save_note.",
        on_iteration=on_iteration,
        stream=False,
    )


async def _main() -> None:
    clients = create_shared_llm_clients()
    transcript = DemoTranscriptLog()
    protocol = LoggingTestCopilotProtocol(transcript)
    try:
        agent = create_test_openrouter_agent(
            clients,
            copilot_protocol=protocol,
            on_iteration=transcript.on_iteration,
        )
        _banner("TURN 0 — USER (initial message)")
        print(textwrap.indent(DEMO_USER_MESSAGE.rstrip(), "  "))
        blocks = await agent.run_async(DEMO_USER_MESSAGE)
        _banner("FINAL — Last assistant message (return value of run_async)")
        print(textwrap.indent(_format_final_blocks(blocks), "  "))
        _banner("DONE", char="-")
        print(
            f"  LLM completion rounds: {transcript.llm_round}  "
            f"(expect ≥2 when tools run: at least one tool-call turn + one text turn)"
        )
    finally:
        await clients.aclose()


if __name__ == "__main__":
    asyncio.run(_main())
