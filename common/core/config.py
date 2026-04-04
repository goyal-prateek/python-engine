"""Active service settings for `common` modules.

The host app (e.g. `apps.www`) must call `configure()` with a `CommonServiceSettings`
instance (usually a subclass with app-specific fields). Until then, readers fall back
to `common_settings_from_env()` so standalone scripts keep working.
"""

from __future__ import annotations

from typing import Optional

from common.core.common_settings import CommonServiceSettings, common_settings_from_env

_active: Optional[CommonServiceSettings] = None


def configure(settings: CommonServiceSettings) -> None:
    """Register the process-wide settings object (typically the app config instance)."""
    global _active
    _active = settings


def get_common_settings() -> CommonServiceSettings:
    if _active is not None:
        return _active
    return common_settings_from_env()


class _CommonSettingsProxy:
    __slots__ = ()

    def __getattr__(self, name: str):
        return getattr(get_common_settings(), name)


config = _CommonSettingsProxy()

__all__ = ["CommonServiceSettings", "configure", "config", "get_common_settings"]
