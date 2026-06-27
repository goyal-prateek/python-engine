"""Logfire-backed logging setup; import after app ``configure()`` for correct env."""

from __future__ import annotations

import logging
import os
import sys

import logfire

from common.core.config import get_common_settings
from common.core.constants import Environment


def _environment_label() -> str:
    s = get_common_settings()
    return str(getattr(s, "ENVIRONMENT", os.getenv("ENVIRONMENT", "local")))


def _service_name() -> str:
    entry = os.path.basename(sys.argv[0]) if sys.argv else "python"
    if entry == "run_www.py":
        return "www"
    return os.path.splitext(entry)[0] or "python-engine"


_env = _environment_label().lower()
_send_logfire = _env not in (Environment.LOCAL.value, Environment.TEST.value)

logfire.configure(
    environment=_environment_label(),
    send_to_logfire=_send_logfire,
    # Keep default scrubbing on so secrets/PII are redacted from spans and logs.
    console=logfire.ConsoleOptions(
        min_log_level="debug",
        span_style="show-parents",
        include_timestamps=True,
        verbose=True,
    ),
    service_name=_service_name(),
    # f-string argument inspection can capture sensitive values; restrict to local/test.
    inspect_arguments=not _send_logfire,
    distributed_tracing=True,
)

logfire_handler = logfire.LogfireLoggingHandler(level=logging.INFO)

logging.basicConfig(
    level=logging.INFO,
    format="[%(name)s] %(message)s",
    datefmt="[%X]",
    handlers=[logfire_handler],
)

logger = logging.getLogger("engine")

__all__ = ["logger", "logfire_handler"]
