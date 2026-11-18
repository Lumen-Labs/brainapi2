"""Replay a staged unit after the caller has acquired the brain publication gate."""

import json

from src.constants.data import Observation, StructuredData, TextChunk
from src.constants.embeddings import Vector
from src.constants.kg import Node, Predicate
from src.core.saving import ingest_cancellation, ingest_staging
from src.core.saving.ingest_staged_output import discard_outputs, load_output
from src.core.saving.ingest_staging import (
    advance_publication,
    finish_publication,
    next_publication_unit,
    units_to_unpublish,
)


_DATA_OPERATIONS = {
    "save_text_chunk": ("text_chunks", "text_chunk", TextChunk),
    "save_observations": ("observations", "observations", Observation),
    "save_structured_data": ("structured_data", "structured_data", StructuredData),
}
_DATA_TABLES = {
    "text_chunks": "data_text_chunks",
    "observations": "data_observations",
    "structured_data": "data_structured_data",
}
_GRAPH_INDEX_METHODS = {
    "refresh_hub_bridges_for_entities",
    "rebuild_hub_bridge_index",
    "rebuild_topic_index",
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


def _record_matches(stored, document):
    return stored is not None and all(stored.get(key) == value for key, value in document.items())


def _rollback_postgres_data(client, brain_id, table, records):
    with client._brain_connection(brain_id) as conn:
        with conn.cursor() as cur:
            for record in records:
                document = record.model_dump(mode="json")
                cur.execute(f"SELECT document FROM {table} WHERE id = %s", (record.id,))
                row = cur.fetchone()
                stored = None if row is None else row[0]
                if isinstance(stored, str):
                    stored = json.loads(stored)
                if not _record_matches(stored, document):
                    continue
                cur.execute(f"DELETE FROM {table} WHERE id = %s", (record.id,))
        conn.commit()


def rollback_data_operation(client, brain_id, operation):
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
            stored = collection.find_one({"_id": record.id})
            if not _record_matches(stored, document):
                continue
            collection.delete_one({"_id": record.id})
        return
    if callable(getattr(client, "_brain_connection", None)) and name in _DATA_TABLES:
        _rollback_postgres_data(client, brain_id, _DATA_TABLES[name], records)
        return
    raise ValueError("Staged data rollback requires a collection or SQL data client")


def replay_vector_operation(client, brain_id, operation):
    args = operation["args"]
    if operation["method"] == "remove_vectors":
        client.remove_vectors(args["ids"], args["store"], brain_id)
        return
    if operation["method"] != "add_vectors":
        raise ValueError("Unsupported staged vector operation")
    vectors = [Vector.model_validate(item) for item in args["vectors"]]
    upsert = getattr(client, "upsert_staged_vectors", client.add_vectors)
    upsert(vectors, args["store"], brain_id)


def rollback_vector_operation(client, brain_id, operation):
    if operation["method"] != "add_vectors":
        raise ValueError("Unsupported staged vector rollback")
    args = operation["args"]
    ids = [vector["id"] for vector in args["vectors"]]
    if ids:
        client.remove_vectors(ids, args["store"], brain_id)


def replay_graph_operation(client, brain_id, operation):
    method = operation["method"]
    args = operation["args"]
    if method == "add_nodes":
        nodes = [Node.model_validate(node) for node in args["nodes"]]
        result = client.add_nodes(nodes, brain_id, args["identification_params"], args["metadata"])
        if isinstance(result, str):
            raise RuntimeError("Staged graph nodes could not be applied")
    elif method in {"add_relationship", "deprecate_relationship"}:
        getattr(client, method)(
            Node.model_validate(args["subject"]),
            Predicate.model_validate(args["predicate"]),
            Node.model_validate(args["to_object"]),
            brain_id,
        )
    elif method == "update_properties":
        client.update_properties(
            args["uuid"], args["updating"], brain_id,
            args["new_properties"], args["properties_to_remove"],
        )
    elif method == "refresh_hub_bridges_for_entities":
        refresh = getattr(client, method, None)
        if refresh is None:
            client.rebuild_hub_bridge_index(brain_id)
        else:
            refresh(brain_id, args["entity_uuids"])
    elif method in {"rebuild_hub_bridge_index", "rebuild_topic_index"}:
        rebuild = getattr(client, method, None)
        if rebuild is not None:
            rebuild(brain_id)
    else:
        raise ValueError("Unsupported staged graph operation")


def rollback_graph_operation(client, brain_id, operation):
    method = operation["method"]
    args = operation["args"]
    if method in _GRAPH_INDEX_METHODS:
        return
    if method == "add_nodes":
        remove_nodes = getattr(client, "remove_nodes", None)
        if not callable(remove_nodes):
            raise ValueError("Staged graph node rollback requires remove_nodes")
        nodes = args.get("nodes") or []
        uuids = [node["uuid"] for node in nodes if node.get("uuid")]
        if len(uuids) != len(nodes):
            raise ValueError("Staged graph nodes cannot be rolled back without uuids")
        if uuids:
            remove_nodes(uuids, brain_id)
        return
    if method == "add_relationship":
        remove_relationships = getattr(client, "remove_relationships", None)
        if not callable(remove_relationships):
            raise ValueError("Staged graph relationship rollback requires remove_relationships")
        remove_relationships(
            [(args["subject"], args["predicate"], args["to_object"])],
            brain_id,
        )
        return
    raise ValueError(f"Unsupported staged graph rollback {method}")


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



def publish_staged_run(brain_id, root_id, acquire_gate, data_client, vector_client, graph_client=None):
    """Apply staged units while the caller's exclusive brain gate is held."""
    def replay(target, operation_id, operation):
        adapter = operation["adapter"]
        if adapter == "data":
            replay_data_operation(data_client, target, operation)
        elif adapter == "vector":
            replay_vector_operation(vector_client, target, operation)
        elif adapter == "graph" and graph_client is not None:
            replay_graph_operation(graph_client, target, operation)
        else:
            raise ValueError(f"Unsupported staged operation {operation_id}")

    with acquire_gate(brain_id, root_id):
        while publish_next_unit(brain_id, root_id, replay):
            pass
        run = ingest_cancellation._runs(brain_id).find_one({"_id": root_id})
        if not run or run["status"] != "publishing":
            return
        finished = finish_publication(brain_id, root_id)
        if finished is None:
            return
        discard_outputs(brain_id, root_id)
        ingest_staging.mark_cleanup_complete(brain_id, root_id)
        return finished


def rollback_published_run(brain_id, root_id, data_client, vector_client, graph_client=None):
    def undo(operation):
        adapter = operation["adapter"]
        if adapter == "data":
            rollback_data_operation(data_client, brain_id, operation)
        elif adapter == "vector" and vector_client is not None:
            rollback_vector_operation(vector_client, brain_id, operation)
        elif adapter == "graph" and graph_client is not None:
            rollback_graph_operation(graph_client, brain_id, operation)
        else:
            raise ValueError(f"Unsupported staged rollback {adapter}")

    for unit in reversed(units_to_unpublish(brain_id, root_id)):
        operations = load_output(brain_id, root_id, unit["output_ref"])
        for operation in reversed(operations):
            undo(operation)
