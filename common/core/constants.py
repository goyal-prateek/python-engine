from enum import StrEnum


class Environment(StrEnum):
    """Environment labels used across apps and logging."""

    LOCAL = "local"
    TEST = "test"
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
