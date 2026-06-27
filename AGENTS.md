# AGENTS.md — python-engine

Guidance for human and AI contributors. This repository is organized for **maintainability**, **clear boundaries**, and **SOLID-oriented** design. When you add features, extend existing patterns instead of introducing parallel ones.

## Purpose

Backend engine exposing a FastAPI app (`apps/www`) that orchestrates LLM text generation, TTS, storage (S3), and optional MongoDB. Shared logic lives under `common/` so future apps can reuse it without duplicating providers or pipelines.

## Repository layout

| Area | Role |
|------|------|
| `run_www.py` | Entry point: runs Uvicorn with `apps.www.app.main:app`, `host`, `port`, and reload from config. |
| `apps/www/` | FastAPI application: `main.py`, `core/config.py`, routers, Pydantic request/response models, thin **services** that compose `common/`. |
| `common/` | Shared modules: LLM/TTS **providers**, **LLM pipelines** (`LLMFunction`), **agents**, prompts, S3, optional **Mongo** (`common.core.db.mongo`). **`common.core.interfaces`** holds **`typing.Protocol`** ports (e.g. persistence); **`common.app.modules.db`** holds Mongo adapters that implement those ports. **`common.core.common_settings`** defines **`CommonServiceSettings`** (keys/AWS/TTS/Mongo fields `common` needs). **`common.core.config`** exposes **`configure()`** + a **`config`** proxy so libraries read the active app’s settings (or env-only defaults for scripts). |
| `common/core/interfaces/` | **Ports** — `Protocol` definitions for cross-cutting contracts (e.g. `PersistentStoreHealth` under `repository/`). Implementations live under `common/app/modules/`, not here. |
| `common/core/db/` | **Infra factories** — e.g. `mongo/` with `create_mongo_app()` and `MongoApp` (async client + DB handle, `ping` / `aclose`). No connections at import time. |
| `common/app/modules/db/` | **Adapters** — Mongo-specific classes that implement `common.core.interfaces` ports (e.g. `MongoPersistentStoreHealth`). Add repositories and collection helpers here. |
| `pyproject.toml` / `uv.lock` | **uv** workspace at repo root (`tool.uv.workspace` members: `common`, `apps/www`). Commit `uv.lock`. |
| `common/pyproject.toml` | Installable package **`common`** (maps `app/` → `common.app`, `core/` → `common.core`) and its direct runtime deps (LLM/TTS/S3 SDKs, pydantic, httpx, …). |
| `apps/www/pyproject.toml` | **www**-only deps (FastAPI, uvicorn, dotenv, …) plus workspace link to `common`. Add future deployable apps as `apps/<name>/pyproject.toml` and a new workspace member. |
| `.github/workflows/aws.yml` | Deploy on push to `main`: `uv sync --frozen --package www`, restart `fastapi-www` on the self-hosted runner. |
| `.pre-commit-config.yaml` | **pre-commit** hooks: trailing whitespace, EOF, YAML/Toml sanity, merge-conflict markers, **Ruff** check (with safe fixes) + **Ruff format**. |
| Root `[tool.ruff]` in **`pyproject.toml`** | **Ruff** lint + format (Python 3.11, import sorting with **`common`** / **`apps`** as first-party). |

**Import convention:** Application code imports `common` as a top-level package (e.g. `from common.app.modules.llm.providers import LLMProvider`). **`common` is installed by uv** into the venv; **`apps.www` is not** — keep the repository root on **`PYTHONPATH`** (e.g. `PYTHONPATH=. uv run --package www python run_www.py`) so `apps.www.*` resolves.

## Linting and formatting

- **Dev deps:** Root **`[dependency-groups] dev`** includes **`ruff`** and **`pre-commit`** (`uv sync --group dev` from the repo root).
- **Ruff:** `uv run ruff check common apps run_www.py` and `uv run ruff format common apps run_www.py`. Rules live under **`[tool.ruff]`** in the root **`pyproject.toml`**; **`common/app/modules/agents/protocol.py`** ignores **B027** for intentional no-op hooks on **`CopilotProtocol`**.
- **pre-commit:** After syncing dev deps, run **`uv run pre-commit install`** once. Use **`uv run pre-commit run --all-files`** before pushing if you want the same checks as CI-style local runs. **Pyright** is not in pre-commit by default (keeps hooks fast); run it manually when you change types (see **Models and typing**).

## Configuration

- **Split:** **`common/core/common_settings.py`** — dataclass **`CommonServiceSettings`** + **`common_settings_from_env()`** (dotenv + env vars) for everything **`common`** reads (API keys, AWS, etc.). **`common/core/config.py`** — **`configure(settings: CommonServiceSettings)`** and **`config`** proxy / **`get_common_settings()`**; until `configure()` runs, callers get **`common_settings_from_env()`** (standalone scripts).
- **Per app:** **`apps/<app>/core/config.py`** defines app-specific subclasses (e.g. **`WwwLocalConfig(CommonServiceSettings)`** with `PORT`, `SERVICE_ROUTE_PREFIX`, `HOT_RELOAD`, …), builds **`config`**, and calls **`common.core.config.configure(config)`** so shared code sees the same object. **Import `apps.<app>.core.config` first** in that app’s `main` (see `apps/www/app/main.py`) so routers never read `common.config` before `configure()`.
- **Secrets and env:** Dotenv is triggered from **`common_settings_from_env()`** / app config construction — keep `.env` loading out of duplicate paths.
- **Mongo (optional):** **`MONGO_URI`** (`Optional[str]`, unset or blank = no Mongo) and **`MONGO_DB_NAME`** (default **`python_engine`** if env omits it) live on **`CommonServiceSettings`** and flow into www config via **`asdict(base)`** like other shared fields.

### Protocol vs `abc.ABC` for new contracts

- **LLM / agents:** Keep using **`typing.Protocol`** (and `@runtime_checkable` when you need `isinstance` checks), matching **`CompletionBackend`** and related types under **`common/app/modules/llm/`**.
- **Persistence and other ports:** Define **`Protocol`** types under **`common/core/interfaces/`** (grouped by concern, e.g. **`repository/`**). Use **`abc.ABC` + `@abstractmethod`** only when you need a concrete base class with shared implementation for subclasses; prefer **`Protocol`** for narrow service-facing ports.

## Architectural decisions (keep these stable)

### 1. Layered API vs domain

- **Routers** (`apps/www/app/routers/`): HTTP only — parse/validate with Pydantic models, call services, return responses or streaming types.
- **Services** (`apps/www/app/services/`): Use cases — orchestrate `LLMFunction` runs, TTS, S3, and map domain outcomes to API shapes. Avoid embedding raw HTTP or provider SDK details when a facade exists.
- **Domain / integrations** (`common/app/modules/`): Providers, prompt types, composable LLM steps, storage helpers, Mongo adapters implementing **`common.core.interfaces`** ports.

### 2. Provider pattern (LLM and TTS)

- **LLM (single stack):** All model I/O goes through **`CompletionRouter`** (`common/app/modules/llm/agent_completion/`) built from **`SharedLLMClients`** — same path for **agents** (tool use) and **single-turn** string prompts (`LLMPromptItem` → `AgentCompletionRequest` via `messages_from_llm_prompt_items` / `complete_llm_prompt_items` / `stream_llm_prompt_text_chunks`). **Adding a new LLM vendor:** implement a backend + branch in `CompletionRouter`; do not add a parallel `LLMBaseProvider` stack.
- **TTS:** `TTSBaseProvider` is a **Pydantic `BaseModel` subclass** with a discriminated-style `provider` literal and shared nested models (e.g. `TTSVoice`, `TTSAudioConfig`).
- **Contract (TTS):** Async methods that must be implemented by each vendor are declared on the base with a body of `...` (ellipsis). Concrete classes override with real implementations. When stricter enforcement is needed, you may introduce `abc.ABC` + `@abstractmethod`, but **keep the same public method names and signatures** as the base so the `TTSProvider` facade stays unchanged.
- **API keys:** LLM keys are read when constructing **`SharedLLMClients`** (`create_shared_llm_clients` uses **`common.core.config.config`**). TTS keys follow the `TTSBaseProvider` / `get_api_key()` pattern. Add new **shared** keys to **`CommonServiceSettings`** and **`common_settings_from_env()`** in **`common/core/common_settings.py`**, then ensure each app’s config subclass carries them (e.g. merge via **`asdict(base)`** as www does). **App-only** settings stay on the app’s config class; import **`apps.<app>.core.config`** where you need those attributes (e.g. route prefixes).

### 3. Facade classes (`LLMProvider`, `TTSProvider`)

- **`LLMProvider`** (`common/app/modules/llm/providers/__init__.py`): Thin wrapper — **`LLMProvider.complete(request, shared_clients)`** delegates to `CompletionRouter`. Use this when you already have an `AgentCompletionRequest` and shared clients; otherwise call `CompletionRouter` or `complete_llm_prompt_items` directly.
- **`TTSProvider`:** `get_provider`, `generate_audio`, `generate_audio_stream`. New TTS backends: subclass `TTSBaseProvider`, register in `get_provider`, extend `TTSProviderLiteral` if needed.

This follows **Dependency Inversion**: call sites depend on `CompletionRouter` / `CompletionBackend`, not on a vendor SDK.

### 4. `LLMFunction` — composable LLM pipelines

Defined in `common/app/modules/llm/functions/__init__.py`:

- **`ParamsModel` / `OutputModel`:** Feature-specific Pydantic models subclass these inner classes (see `common/app/modules/llm/functions/story/models.py`).
- **Steps:** `TransformStep` (sync `Callable` producing a dict) and `CompletionStep` (model id, optional `extend_prompt`, optional `agent_model_config`, generation params). Completions use **`CompletionRouter`** via `complete_llm_prompt_items` / `stream_llm_prompt_text_chunks`.
- **Streaming:** If `stream=True`, the **last** step must be a `CompletionStep`; streaming uses **`stream_llm_prompt_text_chunks`** (OpenAI-compatible / OpenRouter only; use `stream=False` for Gemini-only configs).
- **`LLMFunction.run(params, shared_llm_clients=...)`** — **`shared_llm_clients` is required** whenever a `CompletionStep` runs; inject `app.state.shared_llm_clients` from FastAPI.
- **Feature modules:** One folder per capability (e.g. `story/generate/`) defining params, transforms, prompt extension, and a module-level `LLMFunction(...)` instance — keeps **Single Responsibility** and makes new features copy a proven template.

### 5. Prompts

- **`LLMPromptItem`** (`common/app/modules/llm/promtps/`): `role` in `user` | `assistant`, `content` string. Builders should return `list[LLMPromptItem]`.
- **Folder name:** The package is intentionally spelled `promtps` (historical). Do not rename in drive-by changes; a rename should be a dedicated migration (imports, tooling).

### 6. Agent runtime (`common.app.modules.agents`)

- **Purpose:** Multi-turn **tool loops** (same idea as data-collection-engine’s `base_agent`): `Agent` calls a `CompletionBackend` (typically `CompletionRouter` built from `SharedLLMClients`), runs `execute_tools` / `cancel_tools`, and coordinates with the host via `CopilotProtocol` (abort, HITL, notifications). **Subagents:** named spawns require an explicit **`subagents`** allowlist (`SubagentConfig` per `spawn_name`); `get_or_create_spawn` / delegation builtins are disabled when it is empty. **`make_agent_spawnable`**, `Agent.get_agent_as_tool()`, **`builtin_tools`**, and **`get_acting_agent_name()`** (shared-protocol attribution) live under `common.app.modules.agents` (`SPAWN_REGISTRY`, `clear_spawns_for_session` for tests).
- **LLM wiring:** Canonical messages live in `common.app.modules.llm.messages` (`MessageParamModel`, tool blocks, `CompletionMessageModel`). **Gemini** and **OpenAI-compatible (OpenRouter)** tool completions are implemented under `common.app.modules.llm.agent_completion`; **Anthropic** is reserved (`NotImplementedError` until wired). Per-agent routing uses `AgentModelConfig.provider` in `{"gemini", "openai", "anthropic"}` plus `model`, `max_tokens`, etc.
- **Shared clients:** `create_shared_llm_clients()` in `common.app.modules.llm.clients` builds one shared `httpx.AsyncClient` + `AsyncOpenAI` (OpenRouter) + `genai.Client` (Gemini). FastAPI **`lifespan`** sets **`app.state.shared_llm_clients`** (may be `None` if `OPENROUTER_API_KEY` is missing) and **`app.state.mongo`** via **`create_mongo_app()`** ( **`None`** if `MONGO_URI` is unset). Pass **`shared_llm_clients`** into `LLMFunction.run` and **`LLMProvider.complete`** / **`CompletionRouter`**, or inject `CompletionRouter(clients)` into `Agent` for tool use.
- **Env:** `OPENROUTER_API_KEY` (required to construct shared clients), `GOOGLE_API_KEY` (Gemini; empty string is tolerated at client creation but Gemini calls will fail until set).
- **Typing:** Run **Pyright** from the repo root (uses **`[tool.pyright]`** in **`pyproject.toml`**, the workspace **`.venv`**, and **`extraPaths = ["."]`** so **`apps.www`** resolves the same way as local dev):  
  `uv run pyright`  
  Use **`uv sync --all-packages --group dev`** (or an equivalent sync that installs **`www`** and **`common`** into **`.venv`**) so Pyright can resolve **`fastapi`**, **`pydantic`**, and other dependencies under **`apps/`** and **`common/`**.
- **Tests:** `PYTHONPATH=. python -m unittest common.tests.test_agents` (uses `FakeCompletionBackend`, no network). Mongo: `PYTHONPATH=. python -m unittest common.tests.test_mongo` (integration cases skip unless **`MONGO_URI`** is set).

### 7. Models and typing

- **Pydantic v2** for settings on providers, API request/response models, and LLM params/output.
- Prefer **modern built-in generics** (`list[str]`, `dict[str, Any]`) and explicit `Optional` / `Literal` where the API is constrained.
- **Pyright** is in the root **`[dependency-groups] dev`** (`uv sync --group dev`); run **`uv run pyright`** when making non-trivial type changes. Tighten **`typeCheckingMode`** or per-rule overrides under **`[tool.pyright]`** in **`pyproject.toml`** if you want stricter checking.

### 8. Infrastructure

- **S3:** `common/core/s3.py` — `AsyncS3Client` singleton `async_s3_client`, aiobotocore session, presigned URLs via `get_s3_url`.
- **Mongo (optional):** **`create_mongo_app()`** in **`common.core.db.mongo`** returns **`MongoApp | None`**. If **`MONGO_URI`** is unset or whitespace-only, returns **`None`** without raising. **`MongoApp`** holds **`AsyncMongoClient`** + **`AsyncDatabase`**; use **`await mongo.ping()`** and **`await mongo.aclose()`**. **`apps/www/app/main.py`** lifespan creates Mongo alongside **`SharedLLMClients`**, assigns **`app.state.mongo`**, **pings only when Mongo is configured**, and closes on shutdown. **Do not** construct a global Mongo client at module import time.
- **Persistence ports:** Services may depend on **`PersistentStoreHealth`** (or future repository protocols) typed from **`common.core.interfaces.repository`**; wire concrete **`MongoPersistentStoreHealth`** (or repos) in the app layer from **`app.state.mongo`**.
- **CORS:** Currently permissive in `apps/www/app/main.py`. Tighten per environment when exposing beyond trusted clients.

## SOLID checklist for contributors

| Principle | How it shows up here |
|-----------|----------------------|
| **S**ingle responsibility | Router vs service vs provider vs one `LLMFunction` definition per use case. |
| **O**pen/closed | New behavior: new `agent_completion` backend + `CompletionRouter` branch, or new `LLMFunction` / steps — avoid editing unrelated modules. |
| **L**iskov substitution | New `TTSBaseProvider` implementations must honor the same async methods and typing as the base. |
| **I**nterface segregation | `CompletionBackend` vs `AgentStreamSink`; keep TTS sync-bytes vs async iterator paths distinct. |
| **D**ependency inversion | Call `CompletionRouter` / `LLMProvider.complete` / `TTSProvider` from steps and services; vendor SDK code stays under `agent_completion` and TTS providers. Prefer **`Protocol`** ports in **`common.core.interfaces`** for persistence; Mongo implementations live under **`common.app.modules.db`**. |

## What AI / automated contributors should do

1. **Match existing style:** Same import paths, Pydantic patterns, async everywhere for I/O, facades for providers. Run **Ruff** (or **pre-commit**) on touched Python files before finishing a change.
2. **Extend, don’t fork:** New LLM vendor = backend under `llm/agent_completion/` + `CompletionRouter` branch. New TTS vendor = new file under `tts/providers/`, subclass base, register in the facade.
3. **New HTTP features:** New router + request/response models under `apps/www/app/models/api/`, service method, then `common` only if reusable.
4. **Preserve contracts:** Do not remove or rename `LLMProvider.complete`, `CompletionBackend`, or `LLMFunction.run` parameters without updating all callers (services, tests).
5. **Config and secrets:** Extend **`CommonServiceSettings`** / **`common_settings_from_env()`** for anything **`common/`** reads; extend **`apps/<app>/core/config.py`** for app-only fields; call **`configure()`** from the app config module; use **`apps.<app>.core.config`** in that app’s HTTP layer when you need app-specific fields. New persistence: add **`Protocol`** under **`common/core/interfaces/`**, Mongo implementation under **`common/app/modules/db/mongo/`**, factory in **`common/core/db/mongo/`**, wire from **`app.state.mongo`**.
6. **Avoid scope creep:** No unrelated refactors, no renaming `promtps` without an explicit task. **Dependencies:** use **uv** (`pyproject.toml` per workspace member, single `uv.lock`); add app-specific packages under that app’s `pyproject.toml`, shared packages under `common/pyproject.toml`.

## Operational notes

- **Private dev journal:** `dev-journal/` at repo root (gitignored) holds `ACTIVITY.md`, `DECISIONS.md`, `DEFERRED.md`, `PROGRESS.md` for local decisions and follow-ups. The Cursor project skill **`dev-journal`** (`.cursor/skills/dev-journal/SKILL.md`) instructs the assistant to maintain these files during substantive work.
- Deploy assumes a **self-hosted** GitHub Actions runner and systemd service **`fastapi-www`** on the target host. The venv is populated with **`uv sync --frozen --package www`** at the repo root; ensure the service sets **`PYTHONPATH`** to the repo root (or equivalent) so `apps.www` imports work.
- Bucket name and keys in services (e.g. story flow) are part of the current product wiring; parameterize or move to config when generalizing.

This document should stay aligned with the code: when you make a structural decision (new app under `apps/`, new integration layer), update **AGENTS.md** in the same change.
