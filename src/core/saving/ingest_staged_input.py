"""Durable file input stored outside the ingestion coordinator document."""

from src.core.saving.pipeline_store import collection

PART_SIZE = 4 * 1024 * 1024


def save_file(brain_id, root_id, content_b64):
    parts = collection(brain_id, "ingest_staged_inputs")
    count = 0
    for offset in range(0, len(content_b64), PART_SIZE):
        parts.update_one(
            {"_id": f"{root_id}:{count}"},
            {"$setOnInsert": {
                "import_id": root_id,
                "ordinal": count,
                "content_b64": content_b64[offset:offset + PART_SIZE],
            }},
            upsert=True,
        )
        count += 1
    return count


def load_file(brain_id, root_id, count):
    parts = collection(brain_id, "ingest_staged_inputs")
    rows = list(parts.find({"import_id": root_id}).sort("ordinal", 1))
    if [row["ordinal"] for row in rows] != list(range(count)):
        raise LookupError("Staged file input is incomplete")
    return "".join(row["content_b64"] for row in rows)


def discard_inputs(brain_id, root_id):
    collection(brain_id, "ingest_staged_inputs").delete_many({"import_id": root_id})
