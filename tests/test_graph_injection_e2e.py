"""Injection-payload suite for graph-reaching API endpoints and graph clients."""

from __future__ import annotations

import os
import threading
from contextlib import contextmanager
from unittest.mock import MagicMock, patch
from urllib.parse import urlparse
from uuid import uuid4

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

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
    "POSTGRES_HOST": "localhost",
    "POSTGRES_PORT": "5432",
    "POSTGRES_USERNAME": "postgres",
    "POSTGRES_PASSWORD": "test-password",
}
for key, value in ENV_DEFAULTS.items():
    os.environ.setdefault(key, value)

from src.adapters.graph import GraphAdapter
from src.constants.kg import Node, Predicate
from src.lib.cypher_safety import InvalidCypherIdentifier
from src.lib.postgresql.graph_store import PostgreSQLGraphStore, _BrainGraph
from src.lib.postgresql.networkx_client import NetworkXGraphClient
from src.services.api.errors import install_error_handlers
from src.services.api.routes.meta import meta_router
from src.services.api.routes.model import model_router
from src.services.api.routes.retrieve import retrieve_router

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

BRAIN_ID = "cypherinj"


def _memory_graph_client() -> NetworkXGraphClient:
    client = NetworkXGraphClient.__new__(NetworkXGraphClient)
    store = PostgreSQLGraphStore.__new__(PostgreSQLGraphStore)
    store._brains = {}
    store._brains_lock = threading.RLock()
    store._schema_ready = set()
    store._search_ready_brains = set()
    store._schema_lock = threading.Lock()
    store._persist_node = MagicMock()
    store._persist_relationship = MagicMock()

    def ensure(brain_id):
        store._brains.setdefault(brain_id, _BrainGraph(brain_id))

    store.ensure_database = ensure
    store.get_brain = lambda brain_id: store._brains[brain_id]
    store._load_brain = lambda brain_id: store._brains.setdefault(
        brain_id, _BrainGraph(brain_id)
    )
    store.invalidate_brain = lambda brain_id: None
    client._store = store
    return client


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


GRAPH_ENDPOINTS = (
    ("GET", "/retrieve/entities"),
    ("GET", "/retrieve/relationships"),
    ("GET", "/retrieve/entities/neighbors"),
    ("GET", "/meta/entity-labels"),
    ("GET", "/meta/relationships-properties"),
    ("GET", "/meta/entity-properties"),
)


def _seed_graph(adapter: GraphAdapter, name: str = "Alice") -> tuple[str, str, str]:
    subject = adapter.add_nodes(
        [Node(uuid="alice-1", name=name, labels=["PERSON"], description="person")],
        BRAIN_ID,
    )[0]
    obj = adapter.add_nodes(
        [Node(uuid="acme-1", name="Acme", labels=["ORG"], description="org")],
        BRAIN_ID,
    )[0]
    predicate = Predicate(uuid="rel-1", name="WORKS_AT", description="works at")
    adapter.add_relationship(subject, predicate, obj, BRAIN_ID)
    return subject.uuid, obj.uuid, predicate.uuid


def test_valid_apostrophe_name_round_trips_on_graph_endpoints():
    client = _memory_graph_client()
    adapter = _adapter_for(client)
    with _patched_graph_adapter(adapter):
        _seed_graph(adapter, name="O'Brien")
        api = TestClient(_app())
        created = api.post(
            "/model/entity",
            json={"name": "O'Brien", "labels": ["PERSON"], "description": "valid"},
        )
        assert created.status_code == 200, created.text
        body = created.json()
        assert body[0]["name"] == "O'Brien"

        listed = api.get("/retrieve/entities", params={"query_text": "O'Brien"})
        assert listed.status_code == 200
        payload = listed.json()
        names = [entity["name"] for entity in payload["entities"]]
        assert "O'Brien" in names

        rels = api.get(
            "/retrieve/relationships",
            params={"query_text": "O'Brien", "query_search_target": "node_name"},
        )
        assert rels.status_code == 200

        labels = api.get("/meta/entity-labels")
        assert labels.status_code == 200
        assert "PERSON" in labels.json()

        rel_types = api.get("/meta/relationships-properties")
        assert rel_types.status_code == 200
        assert "WORKS_AT" in rel_types.json()

        props = api.get("/meta/entity-properties")
        assert props.status_code == 200

        neighbors = api.get("/retrieve/entities/neighbors", params={"uuid": "alice-1"})
        assert neighbors.status_code == 200

        updated = api.put(
            "/model/entity",
            json={"uuid": "alice-1", "new_name": "O'Brien", "new_description": "valid"},
        )
        assert updated.status_code == 200, updated.text
        assert updated.json()["name"] == "O'Brien"


def test_injection_payloads_are_stored_as_literal_names():
    client = _memory_graph_client()
    adapter = _adapter_for(client)
    with _patched_graph_adapter(adapter):
        api = TestClient(_app())
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
            created = response.json()[0]
            assert created["name"] == payload
            listed = api.get("/retrieve/entities", params={"query_text": payload})
            assert listed.status_code == 200
            names = [entity["name"] for entity in listed.json()["entities"]]
            assert payload in names


def test_invalid_labels_and_rel_types_return_400():
    client = _memory_graph_client()
    adapter = _adapter_for(client)
    with _patched_graph_adapter(adapter):
        api = TestClient(_app())
        bad_label = api.post(
            "/model/entity",
            json={"name": "Alice", "labels": ["PERSON`DETACH DELETE"], "description": ""},
        )
        assert bad_label.status_code == 400
        assert bad_label.json()["error"]["code"] == "CYPHER_IDENTIFIER_INVALID"

        created = api.post(
            "/model/entity",
            json={"name": "Alice", "labels": ["PERSON"], "description": ""},
        )
        assert created.status_code == 200
        uuid = created.json()[0]["uuid"]
        other = api.post(
            "/model/entity",
            json={"name": "Acme", "labels": ["ORG"], "description": ""},
        )
        other_uuid = other.json()[0]["uuid"]
        bad_rel = api.post(
            "/model/relationship",
            json={
                "subject_uuid": uuid,
                "object_uuid": other_uuid,
                "predicate_name": "OWNS`; MATCH (n) DETACH DELETE n",
                "predicate_description": "owns",
            },
        )
        assert bad_rel.status_code == 400
        assert bad_rel.json()["error"]["code"] == "CYPHER_IDENTIFIER_INVALID"

        listed = api.get("/retrieve/entities", params={"node_labels": "PERSON`x"})
        assert listed.status_code == 400
        listed_from = api.get(
            "/retrieve/relationships", params={"from_node_labels": "PERSON`x"}
        )
        assert listed_from.status_code == 400


def test_every_graph_endpoint_rejects_identifier_injection_without_500():
    client = _memory_graph_client()
    adapter = _adapter_for(client)
    with _patched_graph_adapter(adapter):
        _seed_graph(adapter)
        api = TestClient(_app())
        for method, path in GRAPH_ENDPOINTS:
            response = getattr(api, method.lower())(
                path,
                params={
                    "query_text": "O'Brien",
                    "node_labels": "PERSON",
                    "uuid": "alice-1",
                },
            )
            assert response.status_code < 500, (path, response.status_code, response.text)


def test_networkx_deprecate_relationship_treats_payload_as_literal_name():
    client = _memory_graph_client()
    adapter = _adapter_for(client)
    payload = "O'Brien'; MATCH (n) DETACH DELETE n"
    subject = adapter.add_nodes(
        [Node(uuid="a", name=payload, labels=["PERSON"])], BRAIN_ID
    )[0]
    obj = adapter.add_nodes(
        [Node(uuid="b", name="Acme", labels=["ORG"])], BRAIN_ID
    )[0]
    predicate = Predicate(uuid="r1", name="WORKS_AT", description="works")
    adapter.add_relationship(subject, predicate, obj, BRAIN_ID)
    deprecated = adapter.deprecate_relationship(
        Node(uuid="a", name=payload, labels=["PERSON"]),
        Predicate(name="WORKS_AT", description=""),
        Node(uuid="b", name="Acme", labels=["ORG"]),
        BRAIN_ID,
    )
    assert deprecated is not None
    with pytest.raises(InvalidCypherIdentifier):
        adapter.deprecate_relationship(
            Node(uuid="a", name=payload, labels=["PERSON`x"]),
            Predicate(name="WORKS_AT", description=""),
            Node(uuid="b", name="Acme", labels=["ORG"]),
            BRAIN_ID,
        )


def _postgres_ready() -> bool:
    uri = os.getenv("PROGRESSIVE_POSTGRES_TEST_URI")
    if not uri:
        return False
    try:
        import psycopg2

        conn = psycopg2.connect(uri)
        conn.close()
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _postgres_ready(), reason="PostgreSQL graph backend not available")
def test_postgres_graph_api_injection_payloads():
    from src.config import config

    uri = os.getenv("PROGRESSIVE_POSTGRES_TEST_URI")
    parsed = urlparse(uri)
    brain_id = "cypherp" + uuid4().hex[:8]
    original = (
        config.postgresql.host,
        config.postgresql.port,
        config.postgresql.username,
        config.postgresql.password,
        config.postgresql.database,
        config.postgresql.system_database,
        config.postgresql.maintenance_database,
    )
    config.postgresql.host = parsed.hostname
    config.postgresql.port = parsed.port or 5432
    config.postgresql.username = parsed.username
    config.postgresql.password = parsed.password
    dbname = (parsed.path or "/postgres").lstrip("/") or "postgres"
    config.postgresql.database = dbname
    config.postgresql.system_database = dbname
    config.postgresql.maintenance_database = dbname

    store = PostgreSQLGraphStore()
    client = NetworkXGraphClient.__new__(NetworkXGraphClient)
    client._store = store
    adapter = _adapter_for(client)
    try:
        with _patched_graph_adapter(adapter):
            api = TestClient(_app(brain_id))
            created = api.post(
                "/model/entity",
                json={
                    "name": "O'Brien",
                    "labels": ["PERSON"],
                    "description": "MATCH (n) DETACH DELETE n",
                    "properties": {"note": "'; MATCH (n) DETACH DELETE n //"},
                },
            )
            assert created.status_code == 200, created.text
            assert created.json()[0]["name"] == "O'Brien"
            listed = api.get("/retrieve/entities", params={"query_text": "O'Brien"})
            assert listed.status_code == 200
            names = [entity["name"] for entity in listed.json()["entities"]]
            assert "O'Brien" in names
            rejected = api.post(
                "/model/entity",
                json={"name": "Alice", "labels": ["PERSON`x"], "description": ""},
            )
            assert rejected.status_code == 400
            for method, path in GRAPH_ENDPOINTS:
                response = getattr(api, method.lower())(
                    path, params={"uuid": created.json()[0]["uuid"]}
                )
                assert response.status_code < 500, path
    finally:
        try:
            from src.lib.postgresql._provisioning import drop_brain_database

            drop_brain_database(brain_id)
        except Exception:
            pass
        (
            config.postgresql.host,
            config.postgresql.port,
            config.postgresql.username,
            config.postgresql.password,
            config.postgresql.database,
            config.postgresql.system_database,
            config.postgresql.maintenance_database,
        ) = original
        try:
            import psycopg2
            from psycopg2 import sql
            from src.lib.postgresql._naming import brain_db_name

            admin = psycopg2.connect(uri)
            admin.autocommit = True
            with admin.cursor() as cur:
                cur.execute(
                    sql.SQL("DROP DATABASE IF EXISTS {}").format(
                        sql.Identifier(brain_db_name(brain_id))
                    )
                )
            admin.close()
        except Exception:
            pass
