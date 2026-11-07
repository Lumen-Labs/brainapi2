"""Brain publication admission for processes sharing the deployment lock volume."""

import fcntl
import hashlib
import os
from contextlib import contextmanager
from pathlib import Path

from src.core.saving.ingest_cancellation import _change
from src.core.saving.pipeline_store import collection

_GATE_ID = "publication"


def _gates(brain_id):
    return collection(brain_id, "ingest_access_gates")


def _update_gate(brain_id, transition):
    return _change(brain_id, _GATE_ID, transition, store=_gates)


class BrainPublicationBusy(Exception):
    """The caller should retry once current operations have drained."""


@contextmanager
def _brain_lock(brain_id, mode):
    directory = Path(os.environ["INGEST_PUBLICATION_LOCK_DIR"])
    lock_name = hashlib.sha256(brain_id.encode()).hexdigest()
    with (directory / lock_name).open("a") as lock:
        try:
            fcntl.flock(lock, mode | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise BrainPublicationBusy from exc
        yield


@contextmanager
def brain_access(brain_id):
    with _brain_lock(brain_id, fcntl.LOCK_SH):
        gate = _gates(brain_id).find_one({"_id": _GATE_ID})
        if gate and gate.get("publication"):
            raise BrainPublicationBusy
        yield


def reserve_publication(brain_id, root_id):
    """Close admission before attempting to acquire exclusive brain access."""
    _gates(brain_id).update_one(
        {"_id": _GATE_ID},
        {"$setOnInsert": {"revision": "", "publication": None}},
        upsert=True,
    )
    gate = _update_gate(
        brain_id,
        lambda row: {"publication": root_id} if row["publication"] is None else None,
    )
    return gate["publication"] == root_id


def release_publication(brain_id, root_id):
    """Reopen admission after the exclusive publisher durably completes."""
    _update_gate(
        brain_id,
        lambda gate: {"publication": None} if gate["publication"] == root_id else None,
    )


@contextmanager
def publication_gate(brain_id, root_id):
    if not reserve_publication(brain_id, root_id):
        raise BrainPublicationBusy
    with _brain_lock(brain_id, fcntl.LOCK_EX):
        yield
        release_publication(brain_id, root_id)


def guard_worker(brain_from_args):
    """Reschedule a task that could not enter the brain publication gate."""
    from functools import wraps

    from src.core.saving.ingest_write_buffer import writes_are_staged

    def decorate(fn):
        @wraps(fn)
        def guarded(task, *args, **kwargs):
            if os.getenv("STAGED_INGESTION_ENABLED") != "1" or writes_are_staged():
                return fn(task, *args, **kwargs)

            brain_id = brain_from_args(*args, **kwargs)
            try:
                with brain_access(brain_id):
                    return fn(task, *args, **kwargs)
            except BrainPublicationBusy as exc:
                raise task.retry(exc=exc, countdown=5, max_retries=17_280) from exc

        return guarded

    return decorate
