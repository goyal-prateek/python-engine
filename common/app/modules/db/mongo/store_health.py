from __future__ import annotations

from common.core.db.mongo import MongoApp


class MongoPersistentStoreHealth:
    """Adapter: :class:`PersistentStoreHealth` backed by :class:`MongoApp`."""

    def __init__(self, mongo: MongoApp) -> None:
        self._mongo = mongo

    async def check_reachable(self) -> None:
        await self._mongo.ping()
