"""Durable admission control for staged ingestion publication."""

from contextlib import contextmanager
from uuid import uuid4

from src.core.saving.ingest_cancellation import _change
from src.core.saving.pipeline_store import collection

_GATE_ID = "publication"


def _gates(brain_id):
    return collection(brain_id, "ingest_access_gates")


def _update_gate(brain_id, transition):
    return _change(brain_id, _GATE_ID, transition, store=_gates)


class BrainPublicationBusy(Exception):
    """The caller should retry once current operations have drained."""


def _initialize(brain_id):
    _gates(brain_id).update_one(
        {"_id": _GATE_ID},
        {"$setOnInsert": {"revision": "", "readers": [], "publication": None}},
        upsert=True,
    )


@contextmanager
def brain_access(brain_id):
    _initialize(brain_id)
    token = uuid4().hex
    result = _update_gate(
        brain_id,
        lambda gate: {"readers": [*gate["readers"], token]}
        if gate["publication"] is None else None,
    )
    if token not in result["readers"]:
        raise BrainPublicationBusy
    try:
        yield
    finally:
        _update_gate(
            brain_id,
            lambda gate: {"readers": [entry for entry in gate["readers"] if entry != token]},
        )


def reserve_publication(brain_id, root_id):
    """Close admission, returning whether existing operations have drained."""
    _initialize(brain_id)
    gate = _update_gate(
        brain_id,
        lambda row: {"publication": root_id} if row["publication"] is None else None,
    )
    return gate["publication"] == root_id and not gate["readers"]


def release_publication(brain_id, root_id):
    """Reopen admission after the exclusive publisher durably completes."""
    _update_gate(
        brain_id,
        lambda gate: {"publication": None} if gate["publication"] == root_id else None,
    )
