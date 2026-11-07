"""Immutable, brain-scoped output for a single staging attempt."""

from uuid import NAMESPACE_URL, uuid5

from src.core.saving.pipeline_store import collection


def save_output(brain_id, root_id, unit_id, attempt, operations):
    """Persist JSON-compatible adapter operations before committing a unit."""
    output_ref = uuid5(
        NAMESPACE_URL, f"brainapi:staged-output:{root_id}:{unit_id}:{attempt}"
    ).hex
    outputs = collection(brain_id, "ingest_staged_outputs")
    document = {
        "import_id": root_id,
        "unit_id": unit_id,
        "attempt": attempt,
        "operations": operations,
    }
    outputs.update_one({"_id": output_ref}, {"$setOnInsert": document}, upsert=True)
    stored = outputs.find_one({"_id": output_ref})
    if stored is None or any(stored.get(key) != value for key, value in document.items()):
        raise ValueError("Staging attempt already has different output")
    return output_ref


def load_output(brain_id, root_id, output_ref):
    outputs = collection(brain_id, "ingest_staged_outputs")
    document = outputs.find_one({"_id": output_ref, "import_id": root_id})
    if document is None:
        raise LookupError("Staged ingestion output not found")
    return document["operations"]
