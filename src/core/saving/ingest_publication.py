"""Replay a staged unit after the caller has acquired the brain publication gate."""

import json
import logging

from src.constants.data import Observation, StructuredData, TextChunk
from src.constants.embeddings import Vector
from src.constants.kg import Node, Predicate
from src.core.saving import ingest_cancellation, ingest_staging
from src.core.saving.ingest_staged_output import discard_outputs, load_output
from src.core.saving.ingest_staging import (
    abort_publish_rollback,
    advance_publication,
    finish_publication,
    graph_touch,
    graph_touches,
    mark_rollback_done,
    next_publication_unit,
    publication_operation_id,
    record_graph_touch,
    rollback_done_ids,
    set_rollback_reason,
    units_to_unpublish,
)


logger = logging.getLogger(__name__)

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
    if method == "mark_text_chunks_superseded":
        args = operation["args"]
        client.mark_text_chunks_superseded(
            args["ids"],
            args["superseded_at"],
            brain_id,
            superseded_by=args.get("superseded_by"),
            valid_to=args.get("valid_to"),
        )
        return
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
    if method == "mark_text_chunks_superseded":
        raise ValueError("Staged supersession has no safe inverse")
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
    from src.lib.postgresql.vectors import string_to_int64

    args = operation["args"]
    ids = []
    for vector in args["vectors"]:
        vector_id = vector["id"]
        ids.append(vector_id if isinstance(vector_id, int) else string_to_int64(str(vector_id)))
    if ids:
        client.remove_vectors(ids, args["store"], brain_id)


class GraphRollbackUnsafe(ValueError):
    def __init__(self, reason):
        super().__init__(reason)
        self.reason = reason


_ENTITY_FIELDS = (
    "name", "description", "polarity", "metadata", "observations", "happened_at", "last_updated",
)


def _jsonable(value):
    return json.loads(json.dumps(value, default=str))


def _entity_snapshot(entity):
    if entity is None:
        return None
    if hasattr(entity, "model_dump"):
        dump = entity.model_dump(mode="json")
    elif isinstance(entity, dict):
        dump = dict(entity)
    else:
        return None
    uuid = dump.get("uuid") or (dump.get("properties") or {}).get("uuid")
    if uuid:
        dump["uuid"] = uuid
    return _jsonable(dump)


def _changed_keys(entity):
    keys = []
    if hasattr(entity, "model_dump"):
        payload = entity.model_dump(mode="json")
    else:
        payload = dict(entity or {})
    for field in _ENTITY_FIELDS:
        if payload.get(field) not in (None, "", [], {}):
            keys.append(field)
    for key in (payload.get("properties") or {}):
        if key != "uuid":
            keys.append(key)
    return list(dict.fromkeys(keys))


def _as_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return []


def _existing_nodes(client, brain_id, uuids):
    found = {}
    wanted = [str(item) for item in uuids if item]
    if not wanted:
        return found
    getter = getattr(client, "get_by_uuids", None)
    nodes = None
    if callable(getter):
        nodes = getter(wanted, brain_id)
    else:
        getter = getattr(client, "get_by_uuid", None)
        if callable(getter):
            nodes = [getter(uuid, brain_id) for uuid in wanted]
    for node in _as_list(nodes):
        if node is None:
            continue
        uuid = getattr(node, "uuid", None) or (getattr(node, "properties", None) or {}).get("uuid")
        if uuid:
            found[str(uuid)] = node
    return found


def _existing_edges(client, brain_id, uuids):
    found = {}
    wanted = [str(item) for item in uuids if item]
    if not wanted:
        return found
    getter = getattr(client, "get_triples_by_uuid", None)
    if not callable(getter):
        return found
    for triple in _as_list(getter(wanted, brain_id)):
        predicate = triple[1]
        uuid = getattr(predicate, "uuid", None) or (getattr(predicate, "properties", None) or {}).get("uuid")
        if uuid:
            found[str(uuid)] = predicate
    return found


def _predicate_uuid(predicate, fallback=""):
    if predicate is None:
        return fallback
    uuid = getattr(predicate, "uuid", None)
    if uuid:
        return str(uuid)
    if isinstance(predicate, dict):
        return str(predicate.get("uuid") or (predicate.get("properties") or {}).get("uuid") or fallback)
    props = getattr(predicate, "properties", None) or {}
    return str(props.get("uuid") or fallback)


def _touches_for_graph_operation(client, brain_id, operation):
    method = operation["method"]
    args = operation["args"]
    if method == "add_nodes":
        nodes = [Node.model_validate(node) for node in args.get("nodes") or []]
        existing = _existing_nodes(client, brain_id, [node.uuid for node in nodes])
        return [
            {
                "kind": "node",
                "uuid": node.uuid,
                "created": node.uuid not in existing,
                "prior": _entity_snapshot(existing.get(node.uuid)),
                "changed": _changed_keys(node),
            }
            for node in nodes
        ]
    if method == "add_relationship":
        predicate = Predicate.model_validate(args["predicate"])
        uuid = _predicate_uuid(predicate)
        existing = _existing_edges(client, brain_id, [uuid])
        return [
            {
                "kind": "edge",
                "uuid": uuid,
                "created": uuid not in existing,
                "prior": _entity_snapshot(existing.get(uuid)),
                "changed": _changed_keys(predicate),
                "subject": args.get("subject"),
                "predicate": args.get("predicate"),
                "to_object": args.get("to_object"),
            }
        ]
    if method == "deprecate_relationship":
        predicate = Predicate.model_validate(args["predicate"])
        uuid = _predicate_uuid(predicate)
        existing = _existing_edges(client, brain_id, [uuid])
        return [
            {
                "kind": "edge",
                "uuid": uuid,
                "created": False,
                "prior": _entity_snapshot(existing.get(uuid)),
                "changed": ["deprecated", "valid_to", "invalid_at"],
            }
        ]
    if method == "update_properties":
        uuid = str(args.get("uuid") or "")
        kind = "edge" if args.get("updating") == "relationship" else "node"
        existing = (
            _existing_edges(client, brain_id, [uuid])
            if kind == "edge"
            else _existing_nodes(client, brain_id, [uuid])
        )
        changed = list((args.get("new_properties") or {}).keys())
        changed.extend(args.get("properties_to_remove") or [])
        return [
            {
                "kind": kind,
                "uuid": uuid,
                "created": False,
                "prior": _entity_snapshot(existing.get(uuid)),
                "changed": list(dict.fromkeys(changed)),
            }
        ]
    if method == "set_fact_verification":
        uuid = str(args.get("fact_id") or "")
        kind = "edge" if args.get("kind") == "relationship" else "node"
        existing = (
            _existing_edges(client, brain_id, [uuid])
            if kind == "edge"
            else _existing_nodes(client, brain_id, [uuid])
        )
        return [
            {
                "kind": kind,
                "uuid": uuid,
                "created": False,
                "prior": _entity_snapshot(existing.get(uuid)),
                "changed": list((args.get("properties") or {}).keys()),
            }
        ]
    return []


def replay_graph_operation(client, brain_id, operation, *, record_touch=None):
    method = operation["method"]
    args = operation["args"]
    if record_touch is not None and method not in _GRAPH_INDEX_METHODS:
        record_touch(_touches_for_graph_operation(client, brain_id, operation))
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
    elif method == "set_fact_verification":
        client.set_fact_verification(
            args["fact_id"],
            brain_id,
            kind=args["kind"],
            properties=args["properties"],
            history_entry=args.get("history_entry"),
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


def _incident_edge_uuids(client, node_uuid, brain_id):
    lister = getattr(type(client), "list_incident_relationship_uuids", None)
    if callable(lister):
        result = lister(client, [node_uuid], brain_id)
        if isinstance(result, dict):
            return [str(item) for item in (result.get(node_uuid) or []) if item]
        if isinstance(result, (list, tuple, set)):
            return [str(item) for item in result if item]
    if getattr(client, "graphdb_type", None) == "neo4j" and callable(getattr(type(client), "_run_query", None)):
        result = client._run_query(
            "MATCH (n) WHERE n['uuid'] = $uuid "
            "OPTIONAL MATCH (n)-[r]-() "
            "RETURN collect(DISTINCT r['uuid']) AS uuids",
            brain_id,
            {"uuid": node_uuid},
        )
        records = getattr(result, "records", None) or []
        if not records:
            return []
        return [str(item) for item in (records[0].get("uuids") or []) if item]
    store = (getattr(client, "__dict__", None) or {}).get("_store")
    get_brain = getattr(store, "get_brain", None) if store is not None else None
    if not callable(get_brain):
        return None
    brain = get_brain(brain_id)
    edges = getattr(getattr(brain, "graph", None), "edges", None)
    if not callable(edges):
        return None
    found = []
    for source, target, key, data in edges(keys=True, data=True):
        if source == node_uuid or target == node_uuid:
            found.append(str((data or {}).get("uuid") or key))
    return found


def _flat_prior(prior):
    if not prior:
        return {}
    flat = dict(prior.get("properties") or {})
    for field in _ENTITY_FIELDS:
        if field in prior and prior[field] is not None:
            flat[field] = prior[field]
    flat.pop("uuid", None)
    return flat


def _restore_entity(client, brain_id, kind, uuid, prior, changed):
    updater = getattr(client, "update_properties", None)
    if not callable(updater):
        raise GraphRollbackUnsafe("Staged graph property rollback requires update_properties")
    prior_flat = _flat_prior(prior)
    restore = {}
    remove = []
    for key in changed or []:
        if key == "uuid":
            continue
        if key in prior_flat and prior_flat[key] is not None:
            restore[key] = prior_flat[key]
        else:
            remove.append(key)
    updater(uuid, "relationship" if kind == "edge" else "node", brain_id, restore, remove)


def _owned_created_edge_uuids(ledger):
    owned = set()
    for touches in (ledger or {}).values():
        for touch in touches or []:
            if touch.get("kind") == "edge" and touch.get("created") and touch.get("uuid"):
                owned.add(str(touch["uuid"]))
    return owned


def _touch_item_id(touch, operation_id=None):
    kind = touch.get("kind") or "node"
    uuid = touch.get("uuid")
    if operation_id:
        return f"{operation_id}:{kind}:{uuid}"
    return f"{kind}:{uuid}"


def _require_touches(operation, touches):
    method = operation["method"]
    if method in _GRAPH_INDEX_METHODS:
        return []
    if touches:
        return touches
    raise GraphRollbackUnsafe(
        f"Staged graph {method} cannot be rolled back without created/reused markers"
    )


def _precheck_graph_step(client, operation, touches):
    method = operation["method"]
    if method in _GRAPH_INDEX_METHODS:
        return
    marked = _require_touches(operation, touches)
    created = [touch for touch in marked if touch.get("created")]
    reused = [touch for touch in marked if not touch.get("created")]
    if created and method == "add_nodes" and not callable(getattr(client, "remove_nodes", None)):
        raise GraphRollbackUnsafe("Staged graph node rollback requires remove_nodes")
    if created and method == "add_relationship" and not callable(getattr(client, "remove_relationships", None)):
        raise GraphRollbackUnsafe("Staged graph relationship rollback requires remove_relationships")
    if reused and not callable(getattr(client, "update_properties", None)):
        raise GraphRollbackUnsafe("Staged graph property rollback requires update_properties")
    if method == "add_nodes":
        nodes = operation.get("args", {}).get("nodes") or []
        if any(not node.get("uuid") for node in nodes):
            raise GraphRollbackUnsafe("Staged graph nodes cannot be rolled back without uuids")
    if method not in {
        "add_nodes", "add_relationship", "update_properties",
        "set_fact_verification", "deprecate_relationship",
    } | _GRAPH_INDEX_METHODS:
        raise GraphRollbackUnsafe(f"Unsupported staged graph rollback {method}")


def rollback_graph_operation(
    client,
    brain_id,
    operation,
    touches=None,
    *,
    owned_edge_uuids=None,
    done=None,
    mark_done=None,
    operation_id=None,
):
    method = operation["method"]
    args = operation["args"]
    if method in _GRAPH_INDEX_METHODS:
        return []
    marked = _require_touches(operation, touches if touches is not None else operation.get("touches"))
    completed = []
    skipped = set(done or [])

    def finish(item_id):
        completed.append(item_id)
        skipped.add(item_id)
        if mark_done is not None:
            mark_done([item_id])

    def item_id_for(touch):
        return _touch_item_id(touch, operation_id)

    owned = set(owned_edge_uuids or [])
    if method == "add_nodes":
        created = []
        for touch in marked:
            item_id = item_id_for(touch)
            if item_id in skipped:
                continue
            if touch.get("created"):
                created.append(touch)
            else:
                _restore_entity(client, brain_id, "node", touch["uuid"], touch.get("prior"), touch.get("changed"))
                finish(item_id)
        for touch in created:
            item_id = item_id_for(touch)
            incident = _incident_edge_uuids(client, touch["uuid"], brain_id)
            foreign = [
                edge for edge in (incident or [])
                if edge not in owned
            ] if incident is not None else []
            if incident is not None and foreign:
                finish(item_id)
                continue
            remove_nodes = getattr(client, "remove_nodes", None)
            if not callable(remove_nodes):
                raise GraphRollbackUnsafe("Staged graph node rollback requires remove_nodes")
            remove_nodes([touch["uuid"]], brain_id)
            finish(item_id)
        return completed
    if method == "add_relationship":
        created = []
        for touch in marked:
            item_id = item_id_for(touch)
            if item_id in skipped:
                continue
            if touch.get("created"):
                created.append(touch)
            else:
                _restore_entity(client, brain_id, "edge", touch["uuid"], touch.get("prior"), touch.get("changed"))
                finish(item_id)
        if created:
            remove_relationships = getattr(client, "remove_relationships", None)
            if not callable(remove_relationships):
                raise GraphRollbackUnsafe("Staged graph relationship rollback requires remove_relationships")
            for touch in created:
                remove_relationships(
                    [(
                        touch.get("subject") or args["subject"],
                        touch.get("predicate") or args["predicate"],
                        touch.get("to_object") or args["to_object"],
                    )],
                    brain_id,
                )
                finish(item_id_for(touch))
        return completed
    if method in {"update_properties", "set_fact_verification", "deprecate_relationship"}:
        for touch in marked:
            item_id = item_id_for(touch)
            if item_id in skipped:
                continue
            _restore_entity(
                client, brain_id, touch.get("kind") or "node",
                touch["uuid"], touch.get("prior"), touch.get("changed"),
            )
            finish(item_id)
        return completed
    raise GraphRollbackUnsafe(f"Unsupported staged graph rollback {method}")


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


def _conflict_touches(operation):
    if operation.get("adapter") != "graph":
        return None, None
    if operation.get("method") not in {"add_relationship", "deprecate_relationship"}:
        return None, None
    args = operation.get("args") or {}
    predicate = args.get("predicate") or {}
    subject = args.get("subject") or {}
    fact_id = predicate.get("uuid") if isinstance(predicate, dict) else None
    subject_uuid = subject.get("uuid") if isinstance(subject, dict) else None
    return fact_id, subject_uuid


def _collect_prior_conflict_touches(brain_id, root_id, fact_ids, subject_uuids):
    """Add already-published units to ingest conflict scope. True if a read failed."""
    incomplete = False
    for unit in ingest_staging.published_units_before_cursor(brain_id, root_id):
        try:
            prior = load_output(brain_id, root_id, unit["output_ref"])
        except Exception:
            logger.exception(
                "failed to load prior staged output for conflict scope brain_id=%s root_id=%s ref=%s",
                brain_id,
                root_id,
                unit.get("output_ref"),
            )
            incomplete = True
            continue
        for operation in prior:
            fact_id, subject_uuid = _conflict_touches(operation)
            if fact_id:
                fact_ids.append(str(fact_id))
            if subject_uuid:
                subject_uuids.append(str(subject_uuid))
    return incomplete


def publish_staged_run(brain_id, root_id, acquire_gate, data_client, vector_client, graph_client=None):
    """Apply staged units while the caller's exclusive brain gate is held."""
    fact_ids = []
    subject_uuids = []

    def replay(target, operation_id, operation):
        fact_id, subject_uuid = _conflict_touches(operation)
        if fact_id:
            fact_ids.append(str(fact_id))
        if subject_uuid:
            subject_uuids.append(str(subject_uuid))
        adapter = operation["adapter"]
        if adapter == "data":
            replay_data_operation(data_client, target, operation)
        elif adapter == "vector":
            replay_vector_operation(vector_client, target, operation)
        elif adapter == "graph" and graph_client is not None:
            recorded = graph_touch(brain_id, root_id, operation_id)
            def persist(touches):
                record_graph_touch(brain_id, root_id, operation_id, touches)
            replay_graph_operation(
                graph_client, target, operation,
                record_touch=None if recorded is not None else persist,
            )
        else:
            raise ValueError(f"Unsupported staged operation {operation_id}")

    with acquire_gate(brain_id, root_id):
        prior_incomplete = _collect_prior_conflict_touches(
            brain_id, root_id, fact_ids, subject_uuids
        )
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
        if prior_incomplete:
            logger.warning(
                "skipping ingest conflict scan after incomplete prior-unit read brain_id=%s root_id=%s",
                brain_id,
                root_id,
            )
            return finished
        try:
            from src.core.saving.conflicts import maybe_detect_after_ingest

            maybe_detect_after_ingest(brain_id, fact_ids, subject_uuids=subject_uuids)
        except Exception:
            logger.exception(
                "conflict detection after staged publish failed brain_id=%s root_id=%s",
                brain_id,
                root_id,
            )
        return finished


def _rollback_steps(brain_id, root_id):
    ledger = graph_touches(brain_id, root_id)
    steps = []
    for cursor, unit in reversed(list(enumerate(units_to_unpublish(brain_id, root_id)))):
        operations = load_output(brain_id, root_id, unit["output_ref"])
        base = publication_operation_id(root_id, cursor)
        for index, operation in reversed(list(enumerate(operations))):
            operation_id = f"{base}:{index}"
            steps.append((operation, ledger.get(operation_id), operation_id))
    return steps


def rollback_published_run(brain_id, root_id, data_client, vector_client, graph_client=None):
    steps = _rollback_steps(brain_id, root_id)
    ledger = graph_touches(brain_id, root_id)
    owned_edges = _owned_created_edge_uuids(ledger)
    try:
        for operation, touches, _operation_id in steps:
            if operation.get("adapter") != "graph":
                continue
            if graph_client is None:
                raise GraphRollbackUnsafe("Staged graph rollback requires a graph client")
            _precheck_graph_step(graph_client, operation, touches)
    except GraphRollbackUnsafe as exc:
        abort_publish_rollback(brain_id, root_id, str(exc))
        return

    done = set(rollback_done_ids(brain_id, root_id))

    def persist(item_ids):
        mark_rollback_done(brain_id, root_id, item_ids)
        done.update(item_ids)

    for operation, touches, operation_id in steps:
        adapter = operation["adapter"]
        try:
            if adapter == "data":
                if operation_id in done:
                    continue
                rollback_data_operation(data_client, brain_id, operation)
                persist([operation_id])
            elif adapter == "vector" and vector_client is not None:
                if operation_id in done:
                    continue
                rollback_vector_operation(vector_client, brain_id, operation)
                persist([operation_id])
            elif adapter == "graph" and graph_client is not None:
                rollback_graph_operation(
                    graph_client, brain_id, operation, touches,
                    owned_edge_uuids=owned_edges, done=done, mark_done=persist,
                    operation_id=operation_id,
                )
            else:
                raise ValueError(f"Unsupported staged rollback {adapter}")
        except GraphRollbackUnsafe as exc:
            set_rollback_reason(brain_id, root_id, str(exc))
            raise
        except Exception:
            set_rollback_reason(brain_id, root_id, "rollback_apply_failed")
            raise
