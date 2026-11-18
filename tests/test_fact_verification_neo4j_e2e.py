"""Expert verification states against a real Neo4j instance."""

from __future__ import annotations

import os
import time
from contextlib import contextmanager
from unittest.mock import patch
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
}
for key, value in ENV_DEFAULTS.items():
    os.environ.setdefault(key, value)

from src.adapters.graph import GraphAdapter
from src.constants.kg import Node, Predicate
from src.core.verification.facts import REJECTED, UNVERIFIED, VERIFIED, verification_scope
from src.services.api.controllers import retrieve as retrieve_mod
from src.services.api.errors import install_error_handlers
from src.services.api.routes.model import model_router
from src.services.api.routes.retrieve import retrieve_router

BRAIN_ID = "neo4j"


def _adapter_for(client) -> GraphAdapter:
    adapter = GraphAdapter()
    adapter.add_client(client)
    return adapter


@contextmanager
def _patched_graph_adapter(adapter: GraphAdapter):
    patches = [
        patch("src.services.api.controllers.model.graph_adapter", adapter),
        patch("src.services.api.controllers.retrieve.graph_adapter", adapter),
        patch("src.services.api.controllers.verification.graph_adapter", adapter),
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


def test_neo4j_verify_reject_and_retrieval(neo4j_graph):
    adapter, _client, tracked = neo4j_graph
    suffix = uuid4().hex[:8]
    alice_id = f"alice-{suffix}"
    event_id = f"event-{suffix}"
    bob_id = f"bob-{suffix}"
    made_id = f"made-{suffix}"
    targeted_id = f"targeted-{suffix}"
    tracked.extend([alice_id, event_id, bob_id])

    alice = adapter.add_nodes(
        [Node(uuid=alice_id, name="Alice", labels=["PERSON"], description="person")],
        BRAIN_ID,
    )[0]
    event = adapter.add_nodes(
        [Node(uuid=event_id, name="Meeting", labels=["EVENT"], description="event")],
        BRAIN_ID,
    )[0]
    bob = adapter.add_nodes(
        [Node(uuid=bob_id, name="Bob", labels=["PERSON"], description="person")],
        BRAIN_ID,
    )[0]
    adapter.add_relationship(
        alice,
        Predicate(uuid=made_id, name="MADE", description="made", flow_key="flow-1"),
        event,
        BRAIN_ID,
    )
    adapter.add_relationship(
        event,
        Predicate(
            uuid=targeted_id, name="TARGETED", description="targeted", flow_key="flow-1"
        ),
        bob,
        BRAIN_ID,
    )

    with _patched_graph_adapter(adapter):
        api = TestClient(_app(BRAIN_ID))
        verified = api.post(
            f"/model/facts/{made_id}/verify",
            json={
                "verified_by": "analyst-1",
                "verification_source": "doc-1",
                "note": "checked",
            },
        )
        assert verified.status_code == 200, verified.text
        assert verified.json()["verification"]["state"] == VERIFIED
        assert verified.json()["history"][0]["to_state"] == VERIFIED

        rejected = api.post(
            f"/model/facts/{targeted_id}/reject",
            json={"verified_by": "analyst-1", "note": "bad"},
        )
        assert rejected.status_code == 200, rejected.text
        assert rejected.json()["verification"]["state"] == REJECTED

        listed = api.get("/model/facts", params={"state": "verified"})
        assert listed.status_code == 200
        fact_ids = {item["fact_id"] for item in listed.json()["facts"]}
        assert made_id in fact_ids
        assert targeted_id not in fact_ids

        rels = api.get("/retrieve/relationships")
        assert rels.status_code == 200, rels.text
        rel_ids = {
            (item.get("predicate") or {}).get("uuid")
            for item in rels.json()["relationships"]
        }
        assert made_id in rel_ids
        assert targeted_id not in rel_ids

        rels_all = api.get(
            "/retrieve/relationships", params={"include_rejected": True}
        )
        rel_ids_all = {
            (item.get("predicate") or {}).get("uuid")
            for item in rels_all.json()["relationships"]
        }
        assert targeted_id in rel_ids_all

        with verification_scope(include_rejected=False):
            hidden = adapter.get_event_centric_neighbors([alice_id], BRAIN_ID)
        assert hidden == []
        with verification_scope(include_rejected=True):
            shown = adapter.get_event_centric_neighbors([alice_id], BRAIN_ID)
        assert len(shown) == 1
        assert shown[0][1].verification_state == VERIFIED
        assert shown[0][3].verification_state == REJECTED

        event_neighbors = adapter.get_neighbors([event_id], brain_id=BRAIN_ID)
        neighbor_rel_ids = {
            predicate.uuid for predicate, _node in event_neighbors.get(event_id, [])
        }
        assert made_id in neighbor_rel_ids
        assert targeted_id not in neighbor_rel_ids

        traversed = adapter.traverse_graph(
            brain_id=BRAIN_ID, start_uuid=alice_id, depth=2, limit=20
        )
        hop_ids = {hop["predicate"]["uuid"] for hop in traversed["hops"]}
        assert made_id in hop_ids
        assert targeted_id not in hop_ids

        verified_candidate = {
            "triple": (
                Node(uuid=alice_id, name="Alice", labels=["PERSON"]),
                Predicate(
                    uuid=made_id,
                    name="MADE",
                    description="made",
                    properties={
                        "verification_state": VERIFIED,
                        "verified_by": "analyst-1",
                    },
                ),
                Node(uuid=event_id, name="Meeting", labels=["EVENT"]),
                Predicate(uuid="r2", name="TARGETED", description="t"),
                Node(uuid=bob_id, name="Bob", labels=["PERSON"]),
            ),
            "score": 0.4,
            "ppr_mass": 0.0,
        }
        unverified_candidate = {
            "triple": (
                Node(uuid=alice_id, name="Alice", labels=["PERSON"]),
                Predicate(uuid="other", name="MADE", description="made"),
                Node(uuid="e2", name="Other", labels=["EVENT"]),
                Predicate(uuid="r3", name="TARGETED", description="t"),
                Node(uuid="c", name="Carol", labels=["PERSON"]),
            ),
            "score": 0.4,
            "ppr_mass": 0.0,
        }
        ranked = retrieve_mod._rank_facts_with_completeness(
            [unverified_candidate, verified_candidate]
        )
        assert ranked[0]["triple"][1].uuid == made_id

        got = api.get(f"/model/facts/{made_id}")
        assert got.status_code == 200
        assert got.json()["verification"]["verified_by"] == "analyst-1"
        assert got.json()["verification"]["state"] == VERIFIED
        assert len(got.json()["history"]) == 1
