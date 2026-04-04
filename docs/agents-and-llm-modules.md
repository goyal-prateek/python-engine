# Agents and LLM modules — file guide

This document explains how **`common/app/modules/agents`** and **`common/app/modules/llm`** are split, what each file does, and why that structure exists. It is aligned with local decisions recorded in `dev-journal/DECISIONS.md` (ADR-001 through ADR-004, ADR-006), which motivated separating agent orchestration from vendor-specific completion code and shared message types.

## How the system is broken up

| Layer | Package / area | Responsibility |
|-------|------------------|----------------|
| **Orchestration** | `agents/` | Multi-turn loop: call a completion backend, run or cancel tools, merge results into history, talk to the host via `CopilotProtocol`. **No direct vendor SDK imports** in the agent loop itself. |
| **Wire format + parsing** | `llm/messages/` | Canonical Pydantic models for chat blocks (text, tool use, tool results, images, thinking) and assistant completions (`CompletionMessageModel`, `Usage`, `StopReason`). Conversion helpers to OpenAI-style params and Gemini types live here. |
| **Agent-sized completions** | `llm/agent_completion/` | `CompletionBackend` protocol, request envelope, per-provider backends (Gemini, OpenAI-compatible), routing, streaming accumulation, fakes for tests. Depends on `messages/` and `clients/`; **must not import `agents`**. |
| **Shared HTTP/SDK lifecycle** | `llm/clients.py` | One `SharedLLMClients` per process (or FastAPI lifespan): shared `httpx.AsyncClient`, OpenRouter `AsyncOpenAI`, Gemini `genai.Client`. |
| **Prompt pipelines (no tool loop)** | `llm/functions/`, `llm/promtps/` | **`LLMFunction`** composes transforms + **`CompletionStep`**, which turns `LLMPromptItem` lists into **`AgentCompletionRequest`** and calls **`CompletionRouter`** via **`complete_llm_prompt_items`** / **`stream_llm_prompt_text_chunks`**. **`LLMFunction.run(..., shared_llm_clients=...)`** must receive **`SharedLLMClients`** from the app (e.g. `app.state.shared_llm_clients` from FastAPI lifespan). Example: story routes in `apps/www` pass app state into **`StoryService`**; if clients are `None` (missing `OPENROUTER_API_KEY` at startup), the API responds with **503**. |
| **Thin LLM facade** | `llm/providers/` (only `__init__.py`) | **`LLMProvider.complete(request, shared_clients)`** delegates to **`CompletionRouter`** — optional convenience when you already build an `AgentCompletionRequest`. There is no separate OpenRouter provider class anymore; OpenRouter is the **`AsyncOpenAI`** client inside **`SharedLLMClients`**. |

**Significance:** Agents depend on stable types and a small **`CompletionBackend.complete(request) → CompletionMessageModel`** contract (ADR-001). New vendors extend `agent_completion` and `CompletionRouter`; the `Agent` class stays unchanged.

**Import direction:** `agents` → `llm.messages` + `llm.agent_completion`; `llm` does not import `agents` (ADR-004). `agents.tools` implements `ToolSpec` structurally (same methods as the protocol) so `agent_completion` can refer to tools without importing `agents.tools`.

---

## `common/app/modules/agents`

### `__init__.py`

Public package surface: re-exports `Agent`, `MessageHistory`, `CopilotProtocol` (and related types), tool execution helpers, and spawn registry helpers. Consumers should import from here unless they need a specific submodule.

### `agent.py`

- **`Agent`**: Main runtime. Holds `name`, `system`, `session_id`, `CopilotProtocol`, `CompletionBackend`, optional `Tool` list, `AgentModelConfig`, and in-memory `MessageHistory`.
- **`initialize()`**: Bootstraps history with sticky context from the protocol.
- **`_agent_loop` / `run_async`**: Repeatedly builds `AgentCompletionRequest`, calls `completion.complete(...)`, appends the assistant message, handles **`max_tokens` with pending tool calls** (synthetic error tool results + retry hint), executes tools via `execute_tools` or **`cancel_tools`** when unhandled host messages are present, applies a **circuit breaker** on repeated identical tool errors, and stops on HITL, abort, or natural end (no tool calls).
- **`get_agent_as_tool()`**: Wraps the agent as a `Tool` so another agent can call it (requires `description=`).

### `protocol.py`

- **`CopilotProtocol`**: Abstract host-facing hooks — loop lifecycle (`should_continue_loop`, `mark_loop_*`), notifications, unhandled / background message injection, sticky context, streaming and tool callbacks, context usage, compaction, question flows.
- **`CopilotStreamSinkBridge`**: Adapts `CopilotProtocol` to `AgentStreamSink` so OpenAI-compatible streaming backends can fire chunk events into the same host hooks.
- **`NullCopilotProtocol`**: No-op implementation for tests and scripts.
- **`AgentQuestion`**: Minimal shape for ask-user flows (extensible in product code).

### `history.py`

- **`MessageHistory`**: In-memory message list with **best-effort token accounting** from assistant `Usage` (ADR-006 — persistence deferred).
- **`truncate()`**: Drops oldest turns when over `context_window_tokens` (simplified token subtraction).
- **`add_human_in_the_loop_message`**: Merges user follow-up into an existing tool result or appends a synthetic user message.

### `spawnable.py`

- **`SPAWN_REGISTRY`**: Maps `(session_id, parent_agent_name, spawn_name)` → child `Agent` so named sub-agents keep isolated histories while sharing protocol and completion backend.
- **`make_agent_spawnable(agent)`**: Returns tools to run/list spawns; **`clear_spawns_for_session`** clears registry entries for tests or teardown.

### `tools/base.py`

- **`Tool[BaseModelT]`**: Abstract tool with Pydantic `input_model`, JSON schema export for **OpenAI** (`to_openai_tool`) and **Gemini** (`to_gemini_tool`), and `execute` / `cancel`.
- **`ToolResult` / `ImageToolResult`**: Structured return types (errors, HITL breakout, multimodal content).

### `tools/execute.py`

- **`execute_tools`**: Validates tool inputs with Pydantic, runs `execute` with optional timeout, maps exceptions to `ToolResultBlockParamModel` (parallel by default).
- **`cancel_tools`**: Invokes `tool.cancel` when the host preempts a batch (e.g. unhandled messages).

### `tools/__init__.py`

Re-exports tool types and `execute_tools` / `cancel_tools`.

### `test_agent_openrouter.py`

Integration-style tests against the agent loop using **`FakeCompletionBackend`** and a concrete `CopilotProtocol` implementation (no live network when using the fake).

---

## `common/app/modules/llm`

### `clients.py`

- **`SharedLLMClients`**: Dataclass holding shared `httpx.AsyncClient`, OpenRouter `AsyncOpenAI`, and `genai.Client`; **`aclose()`** for shutdown.
- **`create_shared_llm_clients()`**: Factory used from app lifespan; **requires `OPENROUTER_API_KEY`**; Gemini client is constructed with `GOOGLE_API_KEY` (may be empty until used).

### `messages/blocks.py`

Provider-agnostic **content blocks** (`TextBlockParamModel`, `ToolUseBlockParamModel`, `ToolResultBlockParamModel`, image and tool-result sub-blocks, thinking blocks where applicable). Includes serialization to OpenAI chat message parts and Gemini content types, plus **`flatten_messages_to_openai`** and **`messages_from_llm_prompt_items`** (maps `LLMPromptItem` → `MessageParamModel` for single-turn prompt flows).

### `messages/completion.py`

- **`Usage`**, **`StopReason`**, **`CompletionMessageModel`**: Normalized assistant turn: id, content blocks, model, usage, provider tag, stop reason.
- **Factory methods** such as `from_openai_chat_completion` and `from_gemini_message` centralize parsing (including reasoning / thinking and tool calls).
- Internal helpers for usage extraction and malformed tool JSON handling.

### `messages/__init__.py`

Re-exports block types, **`CompletionMessageModel`** / **`Usage`** / **`StopReason`**, **`flatten_messages_to_openai`**, and **`messages_from_llm_prompt_items`** (see `__all__` in that module).

### `agent_completion/backend_protocol.py`

- **`CompletionBackend`**: `Protocol` with `async complete(request, *, stream_sink=None) -> CompletionMessageModel`.

### `agent_completion/request.py`

- **`AgentProvider`**: Type alias `Literal["gemini", "openai", "anthropic"]` for `AgentModelConfig.provider`.
- **`AgentModelConfig`**: Per-turn/provider settings: `provider`, `model`, token limits, temperature, context window, thinking budget, fallback model list, `openrouter_extra_body`.
- **`AgentCompletionRequest`**: Frozen dataclass: messages, system prompt, tools as `ToolSpec`, and config. Helpers: **`from_parts`**, **`from_llm_prompt_items`** (no tools, for string prompt lists).

### `agent_completion/tool_protocol.py`

- **`ToolSpec`**: Structural protocol (name, `input_model`, `description`, `to_openai_tool`, `to_gemini_tool`) so **`agent_completion` does not import `agents.tools`**.

### `agent_completion/router.py`

- **`CompletionRouter`**: Takes `SharedLLMClients`, owns `GeminiAgentBackend` and `OpenAICompatibleAgentBackend`. **`complete`** tries primary `AgentModelConfig` then **`fallback_models`**; dispatches on `provider`; **`anthropic`** raises `NotImplementedError` (ADR-002).

### `agent_completion/gemini_backend.py`

- **`GeminiAgentBackend`**: `generate_content` with tools and optional thinking config; maps response to `CompletionMessageModel`. **Streaming not implemented** — raises if `stream_sink` is set.

### `agent_completion/openai_backend.py`

- **`OpenAICompatibleAgentBackend`**: Builds chat completion kwargs (flattened messages, system, tools, `extra_body` for OpenRouter). Non-streaming: single completion; streaming: delegates to **`openai_chat_stream`**.

### `agent_completion/openai_chat_stream.py`

- **`accumulate_openai_chat_stream`**: Consumes OpenAI-compatible streaming chunks, merges tool call deltas, forwards text/reasoning chunks to **`AgentStreamSink`**, and builds a final **`CompletionMessageModel`** (including usage when available).

### `agent_completion/stream_sink.py`

- **`AgentStreamSink`**: Protocol for streaming callbacks (`on_stream_start`, `on_text_chunk`, `on_thinking_chunk`, etc.) used by OpenAI-compatible streaming.

### `agent_completion/fake.py`

- **`FakeCompletionBackend`**: Test double that returns queued `CompletionMessageModel` instances in order (no streaming).

### `agent_completion/prompt_completion.py`

- **`complete_llm_prompt_items`**: Single non-streaming completion from `Sequence[LLMPromptItem]` via **`CompletionRouter`** (empty `tools` tuple).
- **`stream_llm_prompt_text_chunks`**: Async iterator of text deltas for **`provider="openai"`** (OpenRouter path). Uses an internal queue-backed **`AgentStreamSink`** so chunks can be consumed concurrently with **`CompletionRouter.complete(..., stream_sink=...)`**. Raises **`NotImplementedError`** for non-`openai` providers (Gemini streaming is not implemented for this helper).

### `agent_completion/__init__.py`

Re-exports completion types (see package `__all__`): `CompletionBackend`, `CompletionRouter`, `AgentCompletionRequest`, `AgentModelConfig`, `AgentProvider`, `FakeCompletionBackend`, `AgentStreamSink`, `ToolSpec`, `complete_llm_prompt_items`, `stream_llm_prompt_text_chunks`.

### `providers/__init__.py`

- **`LLMProvider.complete(request, shared_clients)`**: Delegates to **`CompletionRouter(shared_clients).complete(request)`**.

### `functions/__init__.py`

- **`LLMFunction`**: Composable pipeline: **`TransformStep`** (sync `Callable` → dict) and **`CompletionStep`**.
- **`CompletionStep`**: `model`, optional `extend_prompt`, `max_tokens`, `temperature`, and optional **`agent_model_config`**. If **`agent_model_config`** is set, it is deep-copied and used as-is; otherwise **`resolved_model_config()`** builds **`AgentModelConfig`** with **`provider="openai"`**, the step’s **`model`**, **`max_tokens`** defaulting to **8192** when omitted on the step, and **`temperature`** defaulting to **0.7** when omitted — i.e. OpenRouter by default. Non-streaming calls **`complete_llm_prompt_items`**; streaming calls **`stream_llm_prompt_text_chunks`** with the resolved config.
- **`run(params, shared_llm_clients=...)`**: **`shared_llm_clients`** is required whenever a **`CompletionStep`** is present. If **`stream=True`** on the function, the **last** step must be a **`CompletionStep`**; streaming yields text chunks from the OpenAI-compatible path only.

### `functions/story/`

Feature-specific params/output models and a module-level **`LLMFunction`** instance wiring transforms and prompt extension for story generation (`generate/llm_component.py`, `models.py`).

### `promtps/__init__.py`

- **`LLMPromptItem`**: Simple `role` + string `content` messages for **`LLMFunction`** `extend_prompt` builders. (Package directory name `promtps` is historical; see `AGENTS.md`.)

---

## Quick mental model

1. **`Agent`** = loop + history + tools + protocol; **`CompletionBackend`** = one LLM call.
2. **`CompletionRouter`** = pick Gemini vs OpenRouter (OpenAI API) from **`AgentModelConfig.provider`** and **`model`**.
3. **`messages/`** = the lingua franca between loop, backends, prompt helpers, and tests.
4. **`LLMFunction` + `LLMPromptItem`** = composable pipelines on the **same `CompletionRouter`** as agents; inject **`SharedLLMClients`** into **`run`**.
5. **`LLMProvider.complete`** = thin pass-through to **`CompletionRouter.complete`** when you already have an **`AgentCompletionRequest`**.

For operational notes (env vars, mypy command, tests), see **`AGENTS.md`** in the repository root.
