"""Attempt-local buffer for supported ingestion adapter writes."""

from contextlib import contextmanager
from contextvars import ContextVar

_buffer = ContextVar("ingest_write_buffer", default=None)


@contextmanager
def capture_writes(brain_id):
    operations = []
    token = _buffer.set((brain_id, operations))
    try:
        yield operations
    finally:
        _buffer.reset(token)


def capture_data_write(brain_id, method, argument, value):
    scope = _buffer.get()
    if scope is None:
        return False
    scoped_brain, operations = scope
    if scoped_brain != brain_id:
        raise ValueError("Staged ingestion cannot write to another brain")
    serialized = (
        [item.model_dump(mode="json") for item in value]
        if isinstance(value, list)
        else value.model_dump(mode="json")
    )
    operations.append({
        "adapter": "data",
        "method": method,
        "args": {argument: serialized},
    })
    return True
