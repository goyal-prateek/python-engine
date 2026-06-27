"""Lifespan-scoped MongoDB client (async pymongo). No import-time connection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from pymongo import AsyncMongoClient
from pymongo.asynchronous.database import AsyncDatabase
from pymongo.server_api import ServerApi

if TYPE_CHECKING:
    from common.core.common_settings import CommonServiceSettings


@dataclass
class MongoApp:
    """Process-scoped async Mongo client + selected database."""

    client: AsyncMongoClient
    db: AsyncDatabase

    async def ping(self) -> None:
        await self.client.admin.command("ping")

    async def aclose(self) -> None:
        await self.client.close()


def create_mongo_app(
    *,
    settings: CommonServiceSettings | None = None,
) -> MongoApp | None:
    """Build a Mongo app from configured settings, or return None if ``MONGO_URI`` is unset.

    Does not connect until first operation; call :meth:`MongoApp.ping` from app lifespan
    if you want to fail fast when a URI is configured.
    """
    if settings is None:
        from common.core.config import get_common_settings

        settings = get_common_settings()

    uri = (settings.MONGO_URI or "").strip()
    if not uri:
        return None

    client: AsyncMongoClient = AsyncMongoClient(
        uri,
        server_api=ServerApi(version="1", strict=True, deprecation_errors=True),
        serverSelectionTimeoutMS=5000,
    )
    db = client[settings.MONGO_DB_NAME]
    return MongoApp(client=client, db=db)
