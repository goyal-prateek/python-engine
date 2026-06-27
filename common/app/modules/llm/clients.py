"""Process-scoped HTTP and LLM SDK clients (shared connection pools)."""

from __future__ import annotations

from dataclasses import dataclass

import httpx
from google import genai
from openai import AsyncOpenAI

from common.core.config import config


@dataclass
class SharedLLMClients:
    """Long-lived clients for OpenRouter (OpenAI-compatible) and Gemini."""

    httpx_client: httpx.AsyncClient
    openrouter_openai: AsyncOpenAI
    gemini_client: genai.Client

    async def aclose(self) -> None:
        await self.httpx_client.aclose()


def create_shared_llm_clients(
    *,
    httpx_timeout_s: float = 120.0,
) -> SharedLLMClients:
    """Build shared clients. Call once per process or FastAPI lifespan.

    Raises:
        ValueError: If OPENROUTER_API_KEY is missing (required for the OpenAI client).
    """
    or_key = config.OPENROUTER_API_KEY
    if not or_key:
        raise ValueError("OPENROUTER_API_KEY must be set to create SharedLLMClients")

    http = httpx.AsyncClient(
        timeout=httpx.Timeout(httpx_timeout_s),
        limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
    )
    openai_client = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=or_key,
        http_client=http,
    )
    g_key = config.GOOGLE_API_KEY or ""
    gemini = genai.Client(api_key=g_key)
    return SharedLLMClients(
        httpx_client=http,
        openrouter_openai=openai_client,
        gemini_client=gemini,
    )
