# Agents and LLM modules — file guide

This document explains how **`common/app/modules/agents`** and **`common/app/modules/llm`** are split, what each file does, and why that structure exists. It reflects the same layering as **`AGENTS.md`** and, when present locally, `dev-journal/DECISIONS.md` (ADR-001 through ADR-004, ADR-006), which motivated separating agent orchestration from vendor-specific completion code and shared message types.

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

Public package surface: re-exports `Agent`, `AgentToolInput`, `MessageHistory`, `CopilotProtocol` (and related types), **`SubagentConfig`** / **`normalize_subagents`**, **`get_acting_agent_name`**, **`builtin_tool_names`**, **`builtin_system_prompt_fragment`**, **`DELEGATION_BUILTIN_NAMES`**, tool execution helpers, spawn registry helpers, **`builtin_tools`**, **`get_or_create_spawn`**, and **`subagent_runs`** helpers (`start_delegation`, `wait_all`, `wait_any`, `list_runs_for_parent`, `clear_subagent_runs_for_session`, `is_spawn_busy`). Consumers should import from here unless they need a specific submodule.

### `agent.py`

- **`Agent`**: Main runtime. Holds `name`, `system`, `session_id`, `copilot_protocol`, internal `CompletionBackend`, optional `tools`, in-memory `MessageHistory`, optional **`subagents`** (allowlist: only listed spawn names may be created; empty/absent disables all spawning), optional **`enabled_builtin_tools`** (`None` = all builtins returned by **`builtin_tools`**, empty set = none), and optional `description`, `stream`, and `on_iteration` (per-turn hook). If `model_config` is omitted, defaults to **`AgentModelConfig(provider="gemini", model="gemini-2.0-flash")`**. Use **`completion_backend`** (property) to read the injected backend. LLM **`system`** sent to the model is prefixed with static guidance for any built-in tools actually present on **`tools`** (intersected with **`enabled_builtin_tools`** when set).
- **`initialize()`**: Bootstraps history with sticky context from the protocol.
- **`_agent_loop` / `run_async`**: Repeatedly builds `AgentCompletionRequest`, calls `completion.complete(...)` (with `stream_sink` when `stream=True` on the agent), appends the assistant message, handles **`max_tokens` with pending tool calls** (synthetic error tool results + retry hint), executes tools via `execute_tools` or **`cancel_tools`** when unhandled host messages are present, applies a **circuit breaker** on repeated identical tool errors, and stops on HITL, abort, or natural end (no tool calls). **`run_async`** sets **`get_acting_agent_name()`** for the duration of the run (asyncio context) so a shared **`CopilotProtocol`** can attribute hooks; prefer per-subagent **`copilot_protocol`** on **`SubagentConfig`** when you want isolation without context reads. **`run_async`** also accepts optional **`human_in_the_loop_tool_id`** and **`extra_content_blocks`** for follow-up turns.
- **`get_agent_as_tool()`**: Returns **`Tool[AgentToolInput]`** so another agent can call it; requires **`description=`** on **`Agent`**.

### `protocol.py`

- **`CopilotProtocol`**: Abstract host-facing hooks — loop lifecycle (`should_continue_loop`, `mark_loop_*`), notifications, unhandled / background message injection, sticky context, streaming and tool callbacks, context usage, compaction, question flows.
- **`CopilotStreamSinkBridge`**: Adapts `CopilotProtocol` to `AgentStreamSink` so OpenAI-compatible streaming backends can fire chunk events into the same host hooks.
- **`NullCopilotProtocol`**: No-op implementation for tests and scripts.
- **`AgentQuestion`**: Fields `id`, `prompt`, `kind` (`text` | `single_select` | `multi_select`), and `options` (required for select kinds). Used by **`ask_user_question`** and **`on_question_request`**.
- **`on_question_request` / `on_permission_request`**: Optional-style hooks (default no-op) for hosts to render HITL UI when built-in **`ask_user_question`** / **`ask_user_permission`** run. Resume the parent with **`run_async(..., human_in_the_loop_tool_id=<tool_use_id>)`** and a user message (for example JSON answers or `{"allowed": bool, "message": str}`).

### `history.py`

- **`MessageHistory`**: In-memory message list with **best-effort token accounting** from assistant `Usage` (ADR-006 — persistence deferred).
- **`truncate()`**: Drops oldest turns when over `context_window_tokens` (simplified token subtraction).
- **`add_human_in_the_loop_message`**: Merges user follow-up into an existing tool result or appends a synthetic user message.

### `spawnable.py`

- **`SPAWN_REGISTRY`**: Maps `(session_id, parent_agent_name, spawn_name)` → child `Agent` so named sub-agents keep isolated histories while sharing protocol and completion backend (unless overridden per spawn).
- **`get_or_create_spawn(parent, spawn_name)`**: Creates or returns the child agent for that key (bootstraps history once). **`spawn_name` must appear in `parent.subagents`**; if **`subagents` is empty, raises** (spawning is opt-in via allowlist).
- **`make_agent_spawnable(agent)`**: Returns tools to run/list spawns synchronously when **`agent.subagents` is non-empty**; otherwise returns an empty list. If a **`delegate_to_subagent`** run is in flight for the same spawn, the sync tool returns an error so two concurrent **`run_async`** calls cannot corrupt one child history.
- **`clear_spawns_for_session`**: Clears spawn registry entries for tests or teardown (call alongside **`clear_subagent_runs_for_session`** if you use async delegations).

### `subagent_config.py`

- **`SubagentConfig`**: Optional overrides per allowlisted spawn (`system`, `tools`, `agent_model_config`, `description`, `copilot_protocol`, `stream`). Defaults inherit from the parent at spawn creation time.
- **`normalize_subagents`**: Normalizes constructor input (`dict` or sequence of configs with **`spawn_name`** set) to an internal dict.

### `execution_context.py`

- **`get_acting_agent_name`**: Reads the asyncio **`ContextVar`** set during **`Agent.run_async`** for shared-protocol attribution (snapshot early in hooks if you defer work to another task).

### `builtin_prompts.py`

- Stable **`BUILTIN_TOOL_GUIDANCE`** copy and **`builtin_system_prompt_fragment(include=...)`** used by **`Agent`** to prefix **`system`** for built-in tools present on the agent.

### `subagent_runs.py`

- In-process registry of **async** sub-agent runs: **`start_delegation`**, **`wait_all`**, **`wait_any`**, **`list_runs_for_parent`**, **`clear_subagent_runs_for_session`**, **`is_spawn_busy`**. At most one active delegated run per `(session_id, parent_name, spawn_name)`.
- Completed runs keep a short **`result_preview`** and optional **`result_blocks_json`** for **`wait_*`** / listing.

### `tools/builtin.py`

- **`builtin_tools(agent)`** (requires **`description=`** on the agent): includes delegation tools only when **`agent.subagents` is non-empty**; always includes **`ask_user_question`** and **`ask_user_permission`** when not filtered out by **`enabled_builtin_tools`**. Respects **`Agent.enabled_builtin_tools`** (`None` = all tools that are built for that agent; empty set = no builtins). HITL tools set **`ToolResult.break_out_of_loop`** so the agent loop stops until the host resumes with **`human_in_the_loop_tool_id`**.
- **Nested `mark_loop_*`**: Child agents use the same **`copilot_protocol`** as the parent when **`SubagentConfig.copilot_protocol`** is unset; parallel sub-agents can interleave **`mark_loop_as_in_progress` / `mark_loop_as_completed`** on the host. Prefer a dedicated protocol per subagent or **`get_acting_agent_name()`** when sharing one instance.

### `tools/base.py`

- **`Tool[BaseModelT]`**: Abstract tool with Pydantic `input_model`, JSON schema export for **OpenAI** (`to_openai_tool`) and **Gemini** (`to_gemini_tool`), and `execute` / `cancel`.
- **`ToolResult` / `ImageToolResult`**: Structured return types (errors, HITL breakout, multimodal content).

### `tools/execute.py`

- **`execute_tools`**: Validates tool inputs with Pydantic, runs `execute` with optional timeout, maps exceptions to `ToolResultBlockParamModel` (parallel by default).
- **`cancel_tools`**: Invokes `tool.cancel` when the host preempts a batch (e.g. unhandled messages).

### `tools/__init__.py`

Re-exports tool types and `execute_tools` / `cancel_tools`.

### `common/tests/test_builtin_tools.py`

Covers **`start_delegation`** / **`wait_all`**, double-delegate error, and HITL **`ask_user_question`** / **`ask_user_permission`** resume paths with **`FakeCompletionBackend`**.

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

Re-exports block types (including **`MessageParamModel`**, **`MessageContentBlock`**, and nested tool-result part types), **`CompletionMessageModel`** / **`Usage`** / **`StopReason`**, **`flatten_messages_to_openai`**, and **`messages_from_llm_prompt_items`** (see `__all__` in that module).

### `agent_completion/backend_protocol.py`

- **`CompletionBackend`**: `@runtime_checkable` **`Protocol`** with `async complete(request, *, stream_sink=None) -> CompletionMessageModel`.

### `agent_completion/request.py`

- **`AgentProvider`**: Type alias `Literal["gemini", "openai", "anthropic"]` for `AgentModelConfig.provider`.
- **`AgentModelConfig`**: Per-turn/provider settings: `provider`, `model`, token limits, temperature, context window, thinking budget, `preferred_key_cache_id`, nested **`fallback_models`**, and `openrouter_extra_body`.
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
- **`CompletionStep`**: `model`, optional **`extend_prompt`** (sync or async callable producing **`list[LLMPromptItem]`**), `max_tokens`, `temperature`, and optional **`agent_model_config`**. If **`agent_model_config`** is set, it is deep-copied and used as-is; otherwise **`resolved_model_config()`** builds **`AgentModelConfig`** with **`provider="openai"`**, the step’s **`model`**, **`max_tokens`** defaulting to **8192** when omitted on the step, and **`temperature`** defaulting to **0.7** when omitted — i.e. OpenRouter by default. Non-streaming calls **`complete_llm_prompt_items`**; streaming calls **`stream_llm_prompt_text_chunks`** with the resolved config.
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

For operational notes (env vars, Pyright command, tests), see **`AGENTS.md`** in the repository root.
