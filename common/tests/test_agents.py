"""Agent loop tests with FakeCompletionBackend (no network)."""

from __future__ import annotations

import unittest

from pydantic import BaseModel, Field

from common.app.modules.agents import (
    Agent,
    NullCopilotProtocol,
    SubagentConfig,
    Tool,
    clear_spawns_for_session,
    get_acting_agent_name,
    get_or_create_spawn,
    make_agent_spawnable,
)
from common.app.modules.llm.agent_completion import (
    AgentModelConfig,
    FakeCompletionBackend,
)
from common.app.modules.llm.messages import (
    CompletionMessageModel,
    MessageParamModel,
    TextBlockParamModel,
    ToolUseBlockParamModel,
    Usage,
)


class EchoInput(BaseModel):
    """Echo a string."""

    x: str = Field(..., description="Text to echo")


class EchoTool(Tool[EchoInput]):
    def __init__(self) -> None:
        super().__init__(name="echo", input_model=EchoInput)

    async def execute(self, inp: EchoInput) -> str:
        return inp.x


class TestAgentLoop(unittest.IsolatedAsyncioTestCase):
    async def test_tool_then_text(self) -> None:
        first = CompletionMessageModel(
            id="1",
            content=MessageParamModel(
                role="assistant",
                content=[
                    ToolUseBlockParamModel(
                        id="c1",
                        name="echo",
                        input={"x": "hi"},
                        type="tool_use",
                    )
                ],
            ),
            model="test",
            usage=Usage(input_tokens=1, output_tokens=1),
            provider="openai",
            stop_reason="tool_use",
        )
        second = CompletionMessageModel(
            id="2",
            content=MessageParamModel(
                role="assistant",
                content=[TextBlockParamModel(type="text", text="done")],
            ),
            model="test",
            usage=Usage(input_tokens=1, output_tokens=1),
            provider="openai",
            stop_reason="end_turn",
        )
        fake = FakeCompletionBackend()
        fake.enqueue(first)
        fake.enqueue(second)
        agent = Agent(
            name="a",
            system="sys",
            session_id="s1",
            copilot_protocol=NullCopilotProtocol(),
            completion=fake,
            tools=[EchoTool()],
            model_config=AgentModelConfig(provider="openai", model="test/model"),
            description="Test agent",
        )
        await agent.initialize()
        out = await agent.run_async("go")
        self.assertEqual(len(out), 1)
        self.assertIsInstance(out[0], TextBlockParamModel)
        assert isinstance(out[0], TextBlockParamModel)
        self.assertEqual(out[0].text, "done")

    async def test_spawnable_registry(self) -> None:
        clear_spawns_for_session("s2")
        fake = FakeCompletionBackend()
        done = CompletionMessageModel(
            id="d",
            content=MessageParamModel(
                role="assistant",
                content=[TextBlockParamModel(type="text", text="ok")],
            ),
            model="t",
            usage=Usage(),
            provider="openai",
        )
        fake.enqueue(done)
        parent = Agent(
            name="parent",
            system="s",
            session_id="s2",
            copilot_protocol=NullCopilotProtocol(),
            completion=fake,
            tools=[],
            model_config=AgentModelConfig(provider="openai", model="t"),
            description="Parent agent",
            subagents={"default": SubagentConfig()},
        )
        await parent.initialize()
        tools = await make_agent_spawnable(parent)
        self.assertEqual(len(tools), 2)
        clear_spawns_for_session("s2")

    async def test_get_or_create_spawn_requires_allowlist(self) -> None:
        clear_spawns_for_session("s-spawn-gate")
        fake = FakeCompletionBackend()
        parent = Agent(
            name="p",
            system="s",
            session_id="s-spawn-gate",
            copilot_protocol=NullCopilotProtocol(),
            completion=fake,
            tools=[],
            model_config=AgentModelConfig(provider="openai", model="t"),
            description="d",
        )
        with self.assertRaisesRegex(ValueError, "no subagents configured"):
            await get_or_create_spawn(parent, "any")
        parent2 = Agent(
            name="p2",
            system="s",
            session_id="s-spawn-gate",
            copilot_protocol=NullCopilotProtocol(),
            completion=fake,
            tools=[],
            model_config=AgentModelConfig(provider="openai", model="t"),
            description="d",
            subagents={"allowed": SubagentConfig()},
        )
        await parent2.initialize()
        with self.assertRaisesRegex(ValueError, "Unknown spawn_name"):
            await get_or_create_spawn(parent2, "other")
        child = await get_or_create_spawn(parent2, "allowed")
        self.assertEqual(child.name, "p2#allowed")
        clear_spawns_for_session("s-spawn-gate")

    async def test_acting_agent_name_during_run(self) -> None:
        class CaptureCopilot(NullCopilotProtocol):
            def __init__(self) -> None:
                super().__init__()
                self.names: list[str | None] = []

            async def mark_loop_as_in_progress(self) -> None:
                self.names.append(get_acting_agent_name())

        copilot = CaptureCopilot()
        fake = FakeCompletionBackend()
        done = CompletionMessageModel(
            id="d",
            content=MessageParamModel(
                role="assistant",
                content=[TextBlockParamModel(type="text", text="x")],
            ),
            model="t",
            usage=Usage(),
            provider="openai",
        )
        fake.enqueue(done)
        agent = Agent(
            name="alpha",
            system="s",
            session_id="s-ctx",
            copilot_protocol=copilot,
            completion=fake,
            tools=[],
            model_config=AgentModelConfig(provider="openai", model="t"),
            description="d",
        )
        await agent.initialize()
        await agent.run_async("hi")
        self.assertEqual(copilot.names, ["alpha"])

    def test_subagents_accept_list_form(self) -> None:
        fake = FakeCompletionBackend()
        agent = Agent(
            name="p",
            system="parent-sys",
            session_id="s-list",
            copilot_protocol=NullCopilotProtocol(),
            completion=fake,
            tools=[],
            model_config=AgentModelConfig(provider="openai", model="t"),
            description="d",
            subagents=[SubagentConfig(spawn_name="worker", system="worker-sys")],
        )
        self.assertEqual(agent.subagents["worker"].system, "worker-sys")
        self.assertIsNone(agent.subagents["worker"].spawn_name)


if __name__ == "__main__":
    unittest.main()
