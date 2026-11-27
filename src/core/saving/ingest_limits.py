"""Output-token cap for ingest extraction chat models."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from contextvars import ContextVar

_output_cap: ContextVar[int | None] = ContextVar("ingest_output_token_cap", default=None)

_OUTPUT_FIELDS = (
    "max_tokens",
    "max_output_tokens",
    "num_predict",
    "max_completion_tokens",
)


def current_output_cap() -> int | None:
    cap = _output_cap.get()
    if cap is None:
        return None
    return cap if cap > 0 else None


@contextmanager
def bind_output_cap(max_new_tokens: int | None):
    value = int(max_new_tokens or 0)
    token = _output_cap.set(value if value > 0 else None)
    try:
        yield
    finally:
        _output_cap.reset(token)


def limit_output(model):
    cap = current_output_cap()
    if not cap:
        return model
    copy = getattr(model, "model_copy", None)
    if not callable(copy):
        return model
    fields = getattr(type(model), "model_fields", None)
    if not isinstance(fields, Mapping):
        fields = {}
    updates = {}
    for field in _OUTPUT_FIELDS:
        if field in fields:
            updates[field] = cap
    if "model_kwargs" in fields:
        kwargs = dict(getattr(model, "model_kwargs", None) or {})
        kwargs["max_tokens"] = cap
        updates["model_kwargs"] = kwargs
    if not updates:
        return model
    return copy(update=updates)
