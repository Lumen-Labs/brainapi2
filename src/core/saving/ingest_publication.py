"""Replay a staged unit after the caller has acquired the brain publication gate."""

from src.constants.data import Observation, StructuredData, TextChunk
from src.core.saving.ingest_staged_output import load_output
from src.core.saving.ingest_staging import advance_publication, next_publication_unit


_DATA_OPERATIONS = {
    "save_text_chunk": ("text_chunks", "text_chunk", TextChunk),
    "save_observations": ("observations", "observations", Observation),
    "save_structured_data": ("structured_data", "structured_data", StructuredData),
}


def replay_data_operation(client, brain_id, operation):
    """Replay supported data writes using stable document identities."""
    method = operation["method"]
    name, argument, model = _DATA_OPERATIONS[method]
    value = operation["args"][argument]
    records = (
        [model.model_validate(item) for item in value]
        if method == "save_observations"
        else [model.model_validate(value)]
    )
    mongo_collection = getattr(client, "get_collection", None)
    if callable(mongo_collection):
        collection = mongo_collection(name, database=brain_id)
        for record in records:
            document = record.model_dump(mode="json")
            collection.update_one(
                {"_id": record.id}, {"$setOnInsert": document}, upsert=True
            )
            stored = collection.find_one({"_id": record.id})
            if stored is None or any(stored.get(key) != value for key, value in document.items()):
                raise ValueError("Staged data conflicts with an existing record")
        return
    getattr(client, method)(
        brain_id=brain_id,
        **{argument: records if method == "save_observations" else records[0]},
    )


def publish_next_unit(brain_id, root_id, replay_operation):
    """Advance the cursor only after all replay handlers report durable success."""
    unit = next_publication_unit(brain_id, root_id)
    if unit is None:
        return False
    operations = load_output(brain_id, root_id, unit["output_ref"])
    for index, operation in enumerate(operations):
        replay_operation(brain_id, f'{unit["operation_id"]}:{index}', operation)
    advance_publication(brain_id, root_id, unit["cursor"])
    return True
