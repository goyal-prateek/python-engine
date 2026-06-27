"""www application config: shared service fields + www-only settings."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass

from common.core.common_settings import CommonServiceSettings, common_settings_from_env
from common.core.config import configure


@dataclass
class WwwLocalConfig(CommonServiceSettings):
    SERVICE_ROUTE_PREFIX: str = "/www"
    PORT: int = 8000
    HOT_RELOAD: bool = True
    ENVIRONMENT: str = "local"


@dataclass
class WwwDevelopmentConfig(CommonServiceSettings):
    SERVICE_ROUTE_PREFIX: str = "/www"
    PORT: int = 8000
    HOT_RELOAD: bool = False
    ENVIRONMENT: str = "development"


@dataclass
class WwwProductionConfig(CommonServiceSettings):
    SERVICE_ROUTE_PREFIX: str = "/www"
    PORT: int = 8000
    HOT_RELOAD: bool = False
    ENVIRONMENT: str = "production"


def get_config() -> WwwLocalConfig | WwwDevelopmentConfig | WwwProductionConfig:
    base = common_settings_from_env()
    env = str(os.getenv("ENVIRONMENT", "local"))
    if "local" in env:
        env = "local"
    mapping: dict[str, type[WwwLocalConfig | WwwDevelopmentConfig | WwwProductionConfig]] = {
        "local": WwwLocalConfig,
        "development": WwwDevelopmentConfig,
        "production": WwwProductionConfig,
    }
    cls = mapping[env]
    return cls(**asdict(base))


config = get_config()
configure(config)

Config = WwwLocalConfig | WwwDevelopmentConfig | WwwProductionConfig

__all__ = [
    "Config",
    "WwwDevelopmentConfig",
    "WwwLocalConfig",
    "WwwProductionConfig",
    "config",
    "get_config",
]
