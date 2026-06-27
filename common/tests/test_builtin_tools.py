"""Tests for built-in agent tools (subagent runs + HITL)."""

from __future__ import annotations

import unittest

from common.app.modules.agents import (
    Agent,
    NullCopilotProtocol,
    SubagentConfig,
    builtin_tools,
    clear_spawns_for_session,
    clear_subagent_runs_for_session,
    start_delegation,
    wait_all,
)
from common.app.modules.agents.builtin_prompts import builtin_system_prompt_fragment
from common.app.modules.agents.protocol import AgentQuestion
from common.app.modules.llm.agent_completion import AgentModelConfig, FakeCompletionBackend
from common.app.modules.llm.messages import (
    CompletionMessageModel,
    MessageParamModel,
    TextBlockParamModel,
    ToolUseBlockParamModel,
    Usage,
)


def _text_turn(text: str) -> CompletionMessageModel:
    return CompletionMessageModel(
        id="x",
        content=MessageParamModel(
            role="assistant",
            content=[TextBlockParamModel(type="text", text=text)],
        ),
        model="test",
        usage=Usage(),
        provider="openai",
        stop_reason="end_turn",
    )


class RecordingHitlCopilot(NullCopilotProtocol):
    def __init__(self) -> None:
        super().__init__()
        self.question_calls: list[tuple[str, list[AgentQuestion], str | None]] = []
        self.permission_calls: list[tuple[str, str, str, str | None]] = []

    async def on_question_request(
        self,
        tool_id: str,
        questions: list[AgentQuestion],
        context: str | None = None,
    ) -> None:
        self.question_calls.append((tool_id, questions, context))

    async def on_permission_request(
        self,
        tool_id: str,
        title: str,
        body: str,
        context: str | None = None,
    ) -> None:
        self.permission_calls.append((tool_id, title, body, context))


class TestSubagentRuns(unittest.IsolatedAsyncioTestCase):
    async def test_delegate_wait_and_double_delegate_error(self) -> None:
        clear_spawns_for_session("s-delegate")
        clear_subagent_runs_for_session("s-delegate")
        fake = FakeCompletionBackend()
        fake.enqueue(_text_turn("child-done"))
        fake.enqueue(_text_turn("child-done-2"))
        parent = Agent(
            name="parent",
            system="sys",
            session_id="s-delegate",
            copilot_protocol=NullCopilotProtocol(),
            completion=fake,
            tools=[],
            model_config=AgentModelConfig(provider="openai", model="t"),
            description="d",
            subagents={"research": SubagentConfig()},
        )
        await parent.initialize()
        did = await start_delegation(parent, "research", "task-a")
        self.assertTrue(did)
        with self.assertRaises(ValueError):
            await start_delegation(parent, "research", "task-b")
        rows = await wait_all(parent.session_id, parent.name)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["state"], "completed")
        self.assertIn("child-done", rows[0].get("result_preview", ""))
        did2 = await start_delegation(parent, "research", "task-b")
        self.assertNotEqual(did2, did)
        rows2 = await wait_all(parent.session_id, parent.name, only_running=True)
        self.assertEqual(len(rows2), 1)
        self.assertIn("child-done-2", rows2[0].get("result_preview", ""))


class TestBuiltinHitl(unittest.IsolatedAsyncioTestCase):
    async def test_ask_user_question_hitl_resume(self) -> None:
        clear_spawns_for_session("s-hitl")
        clear_subagent_runs_for_session("s-hitl")
        copilot = RecordingHitlCopilot()
        ask_turn = CompletionMessageModel(
            id="1",
            content=MessageParamModel(
                role="assistant",
                content=[
                    ToolUseBlockParamModel(
                        id="tool-hitl-q",
                        name="ask_user_question",
                        type="tool_use",
                        input={
                            "questions": [
                                {
                                    "id": "q1",
                                    "prompt": "Your name?",
                                    "kind": "text",
                                    "options": [],
                                }
                            ],
                            "context": "ctx",
                        },
                    )
                ],
            ),
            model="test",
            usage=Usage(input_tokens=1, output_tokens=1),
            provider="openai",
            stop_reason="tool_use",
        )
        final_turn = _text_turn("acknowledged")
        fake = FakeCompletionBackend()
        fake.enqueue(ask_turn)
        fake.enqueue(final_turn)
        agent = Agent(
            name="a",
            system="sys",
            session_id="s-hitl",
            copilot_protocol=copilot,
            completion=fake,
            tools=[],
            model_config=AgentModelConfig(provider="openai", model="t"),
            description="Host agent",
        )
        agent.tools.extend(builtin_tools(agent))
        await agent.initialize()
        out = await agent.run_async("go")
        self.assertEqual(len(copilot.question_calls), 1)
        self.assertEqual(copilot.question_calls[0][0], "tool-hitl-q")
        self.assertEqual(copilot.question_calls[0][1][0].id, "q1")
        tc = next(b for b in out if isinstance(b, ToolUseBlockParamModel))
        out2 = await agent.run_async('{"q1": "Ada"}', human_in_the_loop_tool_id=tc.id)
        self.assertEqual(len(out2), 1)
        self.assertIsInstance(out2[0], TextBlockParamModel)
        assert isinstance(out2[0], TextBlockParamModel)
        self.assertEqual(out2[0].text, "acknowledged")

    async def test_ask_user_permission_records_call(self) -> None:
        clear_spawns_for_session("s-perm")
        clear_subagent_runs_for_session("s-perm")
        copilot = RecordingHitlCopilot()
        perm_turn = CompletionMessageModel(
            id="1",
            content=MessageParamModel(
                role="assistant",
                content=[
                    ToolUseBlockParamModel(
                        id="tool-hitl-p",
                        name="ask_user_permission",
                        type="tool_use",
                        input={
                            "title": "Delete",
                            "body": "Remove file?",
                            "context": None,
                        },
                    )
                ],
            ),
            model="test",
            usage=Usage(),
            provider="openai",
            stop_reason="tool_use",
        )
        final_turn = _text_turn("proceeding")
        fake = FakeCompletionBackend()
        fake.enqueue(perm_turn)
        fake.enqueue(final_turn)
        agent = Agent(
            name="agent",
            system="sys",
            session_id="s-perm",
            copilot_protocol=copilot,
            completion=fake,
            tools=[],
            model_config=AgentModelConfig(provider="openai", model="t"),
            description="Host agent",
        )
        agent.tools.extend(builtin_tools(agent))
        await agent.initialize()
        out = await agent.run_async("go")
        self.assertEqual(len(copilot.permission_calls), 1)
        self.assertEqual(copilot.permission_calls[0][0], "tool-hitl-p")
        self.assertEqual(copilot.permission_calls[0][1], "Delete")
        tc = next(b for b in out if isinstance(b, ToolUseBlockParamModel))
        out2 = await agent.run_async(
            '{"allowed": true, "message": "ok"}',
            human_in_the_loop_tool_id=tc.id,
        )
        self.assertIsInstance(out2[0], TextBlockParamModel)
        assert isinstance(out2[0], TextBlockParamModel)
        self.assertEqual(out2[0].text, "proceeding")

    def test_builtin_tools_without_subagents_omits_delegation(self) -> None:
        agent = Agent(
            name="a",
            system="s",
            session_id="s1",
            copilot_protocol=NullCopilotProtocol(),
            completion=FakeCompletionBackend(),
            tools=[],
            model_config=AgentModelConfig(provider="openai", model="t"),
            description="d",
        )
        names = {t.name for t in builtin_tools(agent)}
        self.assertIn("ask_user_question", names)
        self.assertIn("ask_user_permission", names)
        self.assertNotIn("delegate_to_subagent", names)

    def test_builtin_tools_with_subagents_includes_delegation(self) -> None:
        agent = Agent(
            name="a",
            system="s",
            session_id="s1",
            copilot_protocol=NullCopilotProtocol(),
            completion=FakeCompletionBackend(),
            tools=[],
            model_config=AgentModelConfig(provider="openai", model="t"),
            description="d",
            subagents={"w": SubagentConfig()},
        )
        names = {t.name for t in builtin_tools(agent)}
        self.assertIn("delegate_to_subagent", names)
        self.assertIn("ask_user_question", names)

    def test_enabled_builtin_tools_empty_returns_no_tools(self) -> None:
        agent = Agent(
            name="a",
            system="s",
            session_id="s1",
            copilot_protocol=NullCopilotProtocol(),
            completion=FakeCompletionBackend(),
            tools=[],
            model_config=AgentModelConfig(provider="openai", model="t"),
            description="d",
            subagents={"w": SubagentConfig()},
            enabled_builtin_tools=frozenset(),
        )
        self.assertEqual(builtin_tools(agent), [])

    def test_builtin_system_prompt_fragment_order(self) -> None:
        frag = builtin_system_prompt_fragment(include=frozenset({"ask_user_question"}))
        self.assertIn("ask_user_question", frag)
        self.assertNotIn("delegate_to_subagent", frag)


if __name__ == "__main__":
    unittest.main()
