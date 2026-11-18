"""Injection-payload suite against a real Neo4j instance.

Skipped locally when Neo4j is unreachable. Required in CI (Quality workflow
starts a Neo4j service and sets NEO4J_E2E_REQUIRED). Community Neo4j only has
the `neo4j` database, which matches production community deployments.
"""

from __future__ import annotations

import os
import time
from contextlib import contextmanager
from unittest.mock import patch
from uuid import uuid4

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from neo4j.exceptions import ClientError

ENV_DEFAULTS = {
    "BRAINPAT_TOKEN": "test-token",
    "MODELS_MODE": "local",
    "EMBEDDINGS_LOCAL_MODEL": "local-model",
    "EMBEDDINGS_SMALL_MODEL": "small-model",
    "EMBEDDING_NODES_DIMENSION": "3",
    "EMBEDDING_TRIPLETS_DIMENSION": "3",
    "EMBEDDING_OBSERVATIONS_DIMENSION": "3",
    "EMBEDDING_DATA_DIMENSION": "3",
    "EMBEDDING_RELATIONSHIPS_DIMENSION": "3",
    "REDIS_HOST": "localhost",
    "REDIS_PORT": "6379",
    "NEO4J_HOST": "localhost",
    "NEO4J_PORT": "7687",
    "NEO4J_USERNAME": "neo4j",
    "NEO4J_PASSWORD": "password",
    "MILVUS_HOST": "localhost",
    "MILVUS_PORT": "19530",
    "MONGO_CONNECTION_STRING": "mongodb://localhost:27017",
    "CELERY_WORKER_CONCURRENCY": "1",
    "OLLAMA_HOST": "localhost",
    "OLLAMA_PORT": "11434",
    "OLLAMA_LLM_SMALL_MODEL": "small",
    "OLLAMA_LLM_LARGE_MODEL": "large",
}
for key, value in ENV_DEFAULTS.items():
    os.environ.setdefault(key, value)

from src.adapters.graph import GraphAdapter
from src.constants.kg import Node, Predicate
from src.lib.cypher_safety import MAX_RAW_QUERY_ROWS, RawCypherRejected
from src.services.api.errors import install_error_handlers
from src.services.api.routes.meta import meta_router
from src.services.api.routes.model import model_router
from src.services.api.routes.retrieve import retrieve_router

BRAIN_ID = "neo4j"
INJECTION_VALUES = [
    "O'Brien",
    "foo}bar",
    "x`y",
    "a//b",
    "DETACH DELETE",
    "MATCH (n) DETACH DELETE n",
    "名前",
    "'; MATCH (n) DETACH DELETE n //",
]
GRAPH_ENDPOINTS = (
    ("GET", "/retrieve/entities"),
    ("GET", "/retrieve/relationships"),
    ("GET", "/retrieve/entities/neighbors"),
    ("GET", "/meta/entity-labels"),
    ("GET", "/meta/relationships-properties"),
    ("GET", "/meta/entity-properties"),
)


def _adapter_for(client) -> GraphAdapter:
    adapter = GraphAdapter()
    adapter.add_client(client)
    return adapter


@contextmanager
def _patched_graph_adapter(adapter: GraphAdapter):
    patches = [
        patch("src.services.api.controllers.model.graph_adapter", adapter),
        patch("src.services.api.controllers.retrieve.graph_adapter", adapter),
        patch("src.services.api.controllers.meta.graph_adapter", adapter),
        patch("src.core.search.entities.graph_adapter", adapter),
        patch("src.core.search.relationships.graph_adapter", adapter),
        patch("src.services.kg_agent.main.graph_adapter", adapter),
    ]
    for item in patches:
        item.start()
    try:
        yield
    finally:
        for item in reversed(patches):
            item.stop()


def _app(brain_id: str = BRAIN_ID) -> FastAPI:
    app = FastAPI()
    install_error_handlers(app)

    @app.middleware("http")
    async def inject_brain(request: Request, call_next):
        request.state.brain_id = brain_id
        return await call_next(request)

    app.include_router(model_router)
    app.include_router(retrieve_router)
    app.include_router(meta_router)
    return app


def _neo4j_required() -> bool:
    flag = os.getenv("NEO4J_E2E_REQUIRED", "")
    return flag.lower() in {"1", "true", "yes"} or os.getenv("CI") == "true"


def _apply_neo4j_env() -> None:
    from src.config import config

    config.neo4j.host = os.getenv("NEO4J_HOST", config.neo4j.host)
    config.neo4j.port = os.getenv("NEO4J_PORT", config.neo4j.port)
    config.neo4j.username = os.getenv("NEO4J_USERNAME", config.neo4j.username)
    config.neo4j.password = os.getenv("NEO4J_PASSWORD", config.neo4j.password)
    config.graph_raw_write_enabled = False


def _neo4j_connectable() -> bool:
    _apply_neo4j_env()
    try:
        from neo4j import GraphDatabase
        from src.config import config

        driver = GraphDatabase.driver(
            f"bolt://{config.neo4j.host}:{config.neo4j.port}",
            auth=(config.neo4j.username, config.neo4j.password),
        )
        driver.verify_connectivity()
        driver.close()
        return True
    except Exception:
        return False


def _require_neo4j() -> None:
    if _neo4j_connectable():
        return
    if _neo4j_required():
        for _ in range(15):
            time.sleep(2)
            if _neo4j_connectable():
                return
        pytest.fail("Neo4j e2e is required but Neo4j is not reachable")
    pytest.skip("Neo4j is not available")


@pytest.fixture
def neo4j_graph():
    _require_neo4j()
    from src.lib.neo4j.client import Neo4jClient

    client = Neo4jClient()
    adapter = _adapter_for(client)
    tracked: list[str] = []
    try:
        yield adapter, client, tracked
    finally:
        if tracked:
            try:
                client._run_query(
                    "MATCH (n) WHERE n['uuid'] IN $uuids DETACH DELETE n",
                    BRAIN_ID,
                    {"uuids": tracked},
                )
            except Exception:
                pass
        try:
            client.driver.close()
        except Exception:
            pass


def _record_value(record, key, default=None):
    if hasattr(record, "get"):
        return record.get(key, default)
    if hasattr(record, "data"):
        return record.data().get(key, default)
    try:
        return record[key]
    except Exception:
        return default


def test_neo4j_injection_payloads_round_trip_every_graph_endpoint(neo4j_graph):
    adapter, client, tracked = neo4j_graph
    with _patched_graph_adapter(adapter):
        api = TestClient(_app(BRAIN_ID))
        created_uuids = []
        for payload in INJECTION_VALUES:
            response = api.post(
                "/model/entity",
                json={
                    "name": payload,
                    "labels": ["PERSON"],
                    "description": payload,
                    "properties": {"note": payload},
                },
            )
            assert response.status_code == 200, response.text
            body = response.json()
            assert body[0]["name"] == payload
            created_uuids.append(body[0]["uuid"])
            tracked.append(body[0]["uuid"])
            listed = api.get("/retrieve/entities", params={"query_text": payload})
            assert listed.status_code == 200
            names = [entity["name"] for entity in listed.json()["entities"]]
            assert payload in names

        other = api.post(
            "/model/entity",
            json={"name": "Acme", "labels": ["ORG"], "description": "org"},
        )
        assert other.status_code == 200, other.text
        other_uuid = other.json()[0]["uuid"]
        tracked.append(other_uuid)
        subject_uuid = created_uuids[0]

        rel = api.post(
            "/model/relationship",
            json={
                "subject_uuid": subject_uuid,
                "object_uuid": other_uuid,
                "predicate_name": "WORKS_AT",
                "predicate_description": INJECTION_VALUES[0],
            },
        )
        assert rel.status_code == 200, rel.text
        rels = api.get(
            "/retrieve/relationships",
            params={"query_text": INJECTION_VALUES[0], "query_search_target": "node_name"},
        )
        assert rels.status_code == 200, rels.text
        rel_names = [
            item["predicate"]["name"] for item in rels.json()["relationships"]
        ]
        assert "WORKS_AT" in rel_names
        rel_lookup = client.execute_operation(
            "MATCH (a)-[r]->(b) WHERE a['uuid'] = '%s' AND b['uuid'] = '%s' "
            "RETURN r['uuid'] AS uuid, type(r) AS t"
            % (subject_uuid, other_uuid),
            BRAIN_ID,
        )
        assert _record_value(rel_lookup.records[0], "t") == "WORKS_AT"
        rel_uuid = _record_value(rel_lookup.records[0], "uuid")
        assert rel_uuid

        updated = api.put(
            "/model/entity",
            json={
                "uuid": subject_uuid,
                "new_name": INJECTION_VALUES[0],
                "new_description": INJECTION_VALUES[1],
                "new_properties": {"note": INJECTION_VALUES[2]},
            },
        )
        assert updated.status_code == 200, updated.text
        assert updated.json()["name"] == INJECTION_VALUES[0]

        neighbors_post = api.post(
            "/retrieve/entities/neighbors",
            json={
                "identification_params": {
                    "name": INJECTION_VALUES[0],
                    "entity_types": ["PERSON"],
                },
                "limit": 10,
            },
        )
        assert neighbors_post.status_code == 200, neighbors_post.text
        assert neighbors_post.json()["main_node"]["uuid"] == subject_uuid

        updated_rel = api.put(
            "/model/relationship",
            json={"uuid": rel_uuid, "new_properties": {"note": INJECTION_VALUES[0]}},
        )
        assert updated_rel.status_code == 200, updated_rel.text

        for method, path in GRAPH_ENDPOINTS:
            response = getattr(api, method.lower())(
                path,
                params={
                    "query_text": INJECTION_VALUES[0],
                    "node_labels": "PERSON",
                    "uuid": subject_uuid,
                },
            )
            assert response.status_code == 200, (
                path,
                response.status_code,
                response.text,
            )

        rejected = api.post(
            "/model/entity",
            json={"name": "Alice", "labels": ["PERSON`x"], "description": ""},
        )
        assert rejected.status_code == 400
        assert rejected.json()["error"]["code"] == "CYPHER_IDENTIFIER_INVALID"

        bad_rel = api.post(
            "/model/relationship",
            json={
                "subject_uuid": subject_uuid,
                "object_uuid": other_uuid,
                "predicate_name": "OWNS`; MATCH (n) DETACH DELETE n",
                "predicate_description": "owns",
            },
        )
        assert bad_rel.status_code == 400


def test_neo4j_deprecate_relationship_treats_payload_as_literal_name(neo4j_graph):
    adapter, client, tracked = neo4j_graph
    payload = "O'Brien'; MATCH (n) DETACH DELETE n"
    subject_uuid = "sub-" + uuid4().hex
    object_uuid = "obj-" + uuid4().hex
    rel_uuid = "rel-" + uuid4().hex
    tracked.extend([subject_uuid, object_uuid])
    subject = adapter.add_nodes(
        [Node(uuid=subject_uuid, name=payload, labels=["PERSON"])],
        BRAIN_ID,
    )[0]
    obj = adapter.add_nodes(
        [Node(uuid=object_uuid, name="Acme", labels=["ORG"])],
        BRAIN_ID,
    )[0]
    predicate = Predicate(uuid=rel_uuid, name="WORKS_AT", description="works")
    adapter.add_relationship(subject, predicate, obj, BRAIN_ID)

    with _patched_graph_adapter(adapter):
        api = TestClient(_app(BRAIN_ID))
        updated = api.put(
            "/model/relationship",
            json={
                "uuid": rel_uuid,
                "new_properties": {"note": payload},
            },
        )
        assert updated.status_code == 200, updated.text

    deprecated = adapter.deprecate_relationship(
        Node(uuid=subject_uuid, name=payload, labels=["PERSON"]),
        Predicate(name="WORKS_AT", description=""),
        Node(uuid=object_uuid, name="Acme", labels=["ORG"]),
        BRAIN_ID,
    )
    assert deprecated is not None

    remaining = client.execute_operation(
        "MATCH (n) WHERE n['uuid'] IN ['%s', '%s'] RETURN count(n) AS c"
        % (subject_uuid, object_uuid),
        BRAIN_ID,
    )
    assert _record_value(remaining.records[0], "c") == 2

    rel_state = client.execute_operation(
        "MATCH ()-[r {uuid: '%s'}]->() RETURN r.deprecated AS deprecated" % rel_uuid,
        BRAIN_ID,
    )
    assert _record_value(rel_state.records[0], "deprecated") is True


def test_neo4j_raw_cypher_rejects_write_delete_and_cross_database(neo4j_graph):
    adapter, client, tracked = neo4j_graph
    probe_uuid = "probe-" + uuid4().hex
    tracked.append(probe_uuid)
    adapter.add_nodes(
        [Node(uuid=probe_uuid, name="KeepMe", labels=["PERSON"])],
        BRAIN_ID,
    )
    before = client.execute_operation(
        "MATCH (n {uuid: '%s'}) RETURN count(n) AS c" % probe_uuid,
        BRAIN_ID,
    )
    assert _record_value(before.records[0], "c") == 1

    with pytest.raises(ClientError) as exc_info:
        client.execute_operation(
            "CREATE (n:Hacked {uuid: '%s'}) RETURN n" % ("hack-" + uuid4().hex),
            BRAIN_ID,
        )
    message = str(exc_info.value).lower()
    assert "read" in message or "access mode" in message or "writing" in message

    with pytest.raises(ClientError):
        client.execute_operation("MATCH (n) DETACH DELETE n", BRAIN_ID)

    after = client.execute_operation(
        "MATCH (n {uuid: '%s'}) RETURN count(n) AS c" % probe_uuid,
        BRAIN_ID,
    )
    assert _record_value(after.records[0], "c") == 1

    adapter_result = adapter.execute_operation(
        "CREATE (n:Hacked {name: 'nope'}) RETURN n",
        BRAIN_ID,
    )
    assert adapter_result.startswith("Error executing graph operation:")

    with pytest.raises(RawCypherRejected):
        client.execute_operation("USE system MATCH (n) RETURN n LIMIT 1", BRAIN_ID)
    with pytest.raises(RawCypherRejected):
        client.execute_operation(
            "CALL dbms.components() YIELD name RETURN name",
            BRAIN_ID,
        )
    with pytest.raises(RawCypherRejected):
        client.execute_operation("CREATE DATABASE stolen", BRAIN_ID)
    with pytest.raises(RawCypherRejected):
        client.execute_operation("CREATE OR REPLACE DATABASE stolen", BRAIN_ID)
    with pytest.raises(RawCypherRejected):
        client.execute_operation(
            "CALL `dbms`.components() YIELD name RETURN name",
            BRAIN_ID,
        )
    with pytest.raises(RawCypherRejected):
        client.execute_operation(
            r"WITH 'a\'' AS s CALL dbms.components() YIELD name RETURN name",
            BRAIN_ID,
        )
    with pytest.raises(RawCypherRejected):
        client.execute_operation(
            "WITH '' AS s CALL dbms.components() YIELD name RETURN name",
            BRAIN_ID,
        )
    with pytest.raises(RawCypherRejected):
        client.execute_operation(
            "CALL dbms/*x*/.components() YIELD name RETURN name",
            BRAIN_ID,
        )
    with pytest.raises(RawCypherRejected):
        client.execute_operation(
            "CALL dbms .components() YIELD name RETURN name",
            BRAIN_ID,
        )

    read_ok = client.execute_operation("RETURN 1 AS n", BRAIN_ID)
    assert _record_value(read_ok.records[0], "n") == 1

    capped = client.execute_operation(
        "UNWIND range(1, 150) AS x RETURN x",
        BRAIN_ID,
    )
    assert len(capped.records) == MAX_RAW_QUERY_ROWS
    assert capped.truncated is True


def test_neo4j_raw_writes_work_only_with_admin_flag(neo4j_graph):
    _adapter, client, tracked = neo4j_graph
    from src.config import config

    write_uuid = "flag-" + uuid4().hex
    tracked.append(write_uuid)
    original = config.graph_raw_write_enabled
    config.graph_raw_write_enabled = True
    try:
        created = client.execute_operation(
            "CREATE (n:Flagged {uuid: '%s'}) RETURN n.uuid AS uuid" % write_uuid,
            BRAIN_ID,
        )
        assert _record_value(created.records[0], "uuid") == write_uuid
    finally:
        config.graph_raw_write_enabled = original

    still_there = client.execute_operation(
        "MATCH (n {uuid: '%s'}) RETURN count(n) AS c" % write_uuid,
        BRAIN_ID,
    )
    assert _record_value(still_there.records[0], "c") == 1
