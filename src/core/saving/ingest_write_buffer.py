"""Attempt-local buffer for supported ingestion adapter writes."""

from contextlib import contextmanager
from contextvars import ContextVar

_buffer = ContextVar("ingest_write_buffer", default=None)


@contextmanager
def capture_writes(brain_id, validate_claim=None):
    operations = []
    token = _buffer.set((brain_id, operations, validate_claim))
    try:
        yield operations
    finally:
        _buffer.reset(token)


def check_staged_claim():
    scope = _buffer.get()
    if scope is not None and scope[2] is not None and not scope[2]():
        from celery.exceptions import Ignore

        raise Ignore()


def writes_are_staged():
    return _buffer.get() is not None


def capture_operation(brain_id, operation):
    scope = _buffer.get()
    if scope is None:
        return False
    scoped_brain, operations, _ = scope
    if scoped_brain != brain_id:
        raise ValueError("Staged ingestion cannot write to another brain")
    operations.append(operation)
    return True


def capture_data_write(brain_id, method, argument, value):
    if _buffer.get() is None:
        return False
    serialized = (
        [item.model_dump(mode="json") for item in value]
        if isinstance(value, list)
        else value.model_dump(mode="json")
    )
    return capture_operation(brain_id, {
        "adapter": "data",
        "method": method,
        "args": {argument: serialized},
    })


def capture_vector_write(brain_id, vectors, store):
    if _buffer.get() is None:
        return False
    return capture_operation(brain_id, {
        "adapter": "vector",
        "method": "add_vectors",
        "args": {"vectors": [vector.model_dump(mode="json") for vector in vectors], "store": store},
    })


def staged_nodes(brain_id, uuids):
    scope = _buffer.get()
    if scope is None or scope[0] != brain_id:
        return {}
    requested = set(uuids)
    return {
        node["uuid"]: node
        for operation in scope[1]
        if operation["adapter"] == "graph" and operation["method"] == "add_nodes"
        for node in operation["args"]["nodes"]
        if node["uuid"] in requested
    }


def staged_vector_records(brain_id, store, ids):
    """Resolve staged inserts and tombstones for requested vector identities."""
    scope = _buffer.get()
    if scope is None or scope[0] != brain_id:
        return {}
    from src.lib.postgresql.vectors import string_to_int64

    requested = {str(vector_id) for vector_id in ids}
    records = {}
    for operation in scope[1]:
        if operation["adapter"] != "vector" or operation["args"]["store"] != store:
            continue
        args = operation["args"]
        if operation["method"] == "add_vectors":
            for vector in args["vectors"]:
                primary = str(string_to_int64(vector["id"]))
                if primary in requested:
                    records[primary] = vector
        elif operation["method"] == "remove_vectors":
            for vector_id in args["ids"]:
                primary = str(vector_id)
                if primary in requested:
                    records[primary] = None
    return records
