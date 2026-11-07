"""Replay a staged unit after the caller has acquired the brain publication gate."""

from src.constants.data import Observation, StructuredData, TextChunk
from src.constants.embeddings import Vector
from src.constants.kg import Node, Predicate
from src.core.saving.ingest_staged_output import discard_outputs, load_output
from src.core.saving.ingest_staging import advance_publication, finish_publication, next_publication_unit


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
        discard_outputs(brain_id, root_id)
        return finish_publication(brain_id, root_id)
