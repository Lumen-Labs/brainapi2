"""Force tracing export off before agent libraries load."""

import os

_TRACING_ENV_VARS = (
    "LANGCHAIN_TRACING_V2",
    "LANGCHAIN_TRACING",
    "LANGSMITH_TRACING",
)


def disable_langchain_tracing() -> None:
    """Overwrite tracing env vars with 'false', including values loaded from .env."""
    for name in _TRACING_ENV_VARS:
        os.environ[name] = "false"


disable_langchain_tracing()
