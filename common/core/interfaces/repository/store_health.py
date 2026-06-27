"""Port for verifying the primary document store is reachable."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class PersistentStoreHealth(Protocol):
    async def check_reachable(self) -> None:
        """Raise if the store cannot be reached."""
        ...
