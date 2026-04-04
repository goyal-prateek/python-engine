"""Agent loop tests with FakeCompletionBackend (no network)."""

from __future__ import annotations

import unittest

from pydantic import BaseModel, Field

from common.app.modules.agents import (
    Agent,
    NullCopilotProtocol,
    Tool,
    clear_spawns_for_session,
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
        )
        await parent.initialize()
        tools = await make_agent_spawnable(parent)
        self.assertEqual(len(tools), 2)
        clear_spawns_for_session("s2")


if __name__ == "__main__":
    unittest.main()
