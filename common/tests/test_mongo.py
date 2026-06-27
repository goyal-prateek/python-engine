"""Mongo factory and persistence port tests (integration optional via MONGO_URI)."""

from __future__ import annotations

import os
import unittest
from dataclasses import replace
from unittest.mock import AsyncMock, MagicMock

from common.app.modules.db.mongo import MongoPersistentStoreHealth
from common.core.common_settings import common_settings_from_env
from common.core.db.mongo import MongoApp, create_mongo_app
from common.core.interfaces.repository import PersistentStoreHealth


class TestCreateMongoApp(unittest.TestCase):
    def test_no_uri_returns_none(self) -> None:
        base = common_settings_from_env()
        s = replace(base, MONGO_URI=None)
        self.assertIsNone(create_mongo_app(settings=s))

    def test_blank_uri_returns_none(self) -> None:
        base = common_settings_from_env()
        s = replace(base, MONGO_URI="   \t")
        self.assertIsNone(create_mongo_app(settings=s))


class TestMongoPersistentStoreHealth(unittest.IsolatedAsyncioTestCase):
    async def test_check_reachable_delegates_to_ping(self) -> None:
        mongo = MagicMock(spec=MongoApp)
        mongo.ping = AsyncMock()
        health = MongoPersistentStoreHealth(mongo)
        await health.check_reachable()
        mongo.ping.assert_awaited_once()
        self.assertIsInstance(health, PersistentStoreHealth)


@unittest.skipUnless(os.getenv("MONGO_URI", "").strip(), "MONGO_URI not set")
class TestMongoIntegration(unittest.IsolatedAsyncioTestCase):
    async def test_ping_and_close(self) -> None:
        app = create_mongo_app()
        self.assertIsNotNone(app)
        assert app is not None
        await app.ping()
        await app.aclose()
