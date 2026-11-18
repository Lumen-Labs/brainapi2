"""Expert verification states against the in-memory PG/NetworkX graph stack."""

from __future__ import annotations

import os
import threading
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

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
from src.core.verification.facts import REJECTED, UNVERIFIED, VERIFIED, verification_scope
from src.lib.postgresql.graph_store import PostgreSQLGraphStore, _BrainGraph
from src.lib.postgresql.networkx_client import NetworkXGraphClient
from src.services.api.controllers import retrieve as retrieve_mod
from src.services.api.errors import install_error_handlers
from src.services.api.routes.model import model_router
from src.services.api.routes.retrieve import retrieve_router

BRAIN_ID = "factverif"


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


def _seed_facts(adapter: GraphAdapter) -> dict[str, str]:
    alice = adapter.add_nodes(
        [Node(uuid="alice-1", name="Alice", labels=["PERSON"], description="person")],
        BRAIN_ID,
    )[0]
    event = adapter.add_nodes(
        [Node(uuid="event-1", name="Meeting", labels=["EVENT"], description="event")],
        BRAIN_ID,
    )[0]
    bob = adapter.add_nodes(
        [Node(uuid="bob-1", name="Bob", labels=["PERSON"], description="person")],
        BRAIN_ID,
    )[0]
    made = Predicate(
        uuid="rel-made",
        name="MADE",
        description="made",
        flow_key="flow-1",
    )
    targeted = Predicate(
        uuid="rel-targeted",
        name="TARGETED",
        description="targeted",
        flow_key="flow-1",
    )
    adapter.add_relationship(alice, made, event, BRAIN_ID)
    adapter.add_relationship(event, targeted, bob, BRAIN_ID)
    return {
        "alice": alice.uuid,
        "event": event.uuid,
        "bob": bob.uuid,
        "made": made.uuid,
        "targeted": targeted.uuid,
    }


def _rel_states(payload: dict) -> dict[str, str]:
    states = {}
    for item in payload.get("relationships") or []:
        predicate = item.get("predicate") or {}
        states[predicate.get("uuid")] = predicate.get("verification_state")
    return states


def test_verify_reject_reset_and_retrieval_respect_state():
    client = _memory_graph_client()
    adapter = _adapter_for(client)
    ids = _seed_facts(adapter)
    with _patched_graph_adapter(adapter):
        api = TestClient(_app())

        listed = api.get("/model/facts")
        assert listed.status_code == 200, listed.text
        body = listed.json()
        assert body["total"] >= 3
        by_id = {item["fact_id"]: item for item in body["facts"]}
        assert by_id[ids["made"]]["verification"]["state"] == UNVERIFIED
        assert by_id[ids["targeted"]]["verification"]["state"] == UNVERIFIED

        missing_actor = api.post(f"/model/facts/{ids['made']}/verify", json={})
        assert missing_actor.status_code == 400

        verified = api.post(
            f"/model/facts/{ids['made']}/verify",
            json={
                "verified_by": "analyst-1",
                "verification_source": "doc-1",
                "note": "source checked",
            },
        )
        assert verified.status_code == 200, verified.text
        verified_body = verified.json()
        assert verified_body["verification"]["state"] == VERIFIED
        assert verified_body["verification"]["verified_by"] == "analyst-1"
        assert verified_body["verification"]["verification_source"] == "doc-1"
        assert len(verified_body["history"]) == 1
        assert verified_body["history"][0]["from_state"] == UNVERIFIED
        assert verified_body["history"][0]["to_state"] == VERIFIED

        rejected = api.post(
            f"/model/facts/{ids['targeted']}/reject",
            json={
                "verified_by": "analyst-1",
                "verification_source": "doc-2",
                "note": "contradicted",
            },
        )
        assert rejected.status_code == 200, rejected.text
        assert rejected.json()["verification"]["state"] == REJECTED

        default_list = api.get("/model/facts")
        listed_ids = {item["fact_id"] for item in default_list.json()["facts"]}
        assert ids["made"] in listed_ids
        assert ids["targeted"] not in listed_ids

        rejected_list = api.get("/model/facts", params={"state": "rejected"})
        assert rejected_list.status_code == 200
        rejected_ids = {item["fact_id"] for item in rejected_list.json()["facts"]}
        assert ids["targeted"] in rejected_ids

        rels = api.get("/retrieve/relationships")
        assert rels.status_code == 200, rels.text
        states = _rel_states(rels.json())
        assert states.get(ids["made"]) == VERIFIED
        assert ids["targeted"] not in states

        rels_all = api.get("/retrieve/relationships", params={"include_rejected": True})
        assert rels_all.status_code == 200
        states_all = _rel_states(rels_all.json())
        assert states_all.get(ids["targeted"]) == REJECTED

        event_neighbors = adapter.get_neighbors([ids["event"]], brain_id=BRAIN_ID)
        neighbor_rel_ids = {
            predicate.uuid
            for predicate, _node in event_neighbors.get(ids["event"], [])
        }
        assert ids["made"] in neighbor_rel_ids
        assert ids["targeted"] not in neighbor_rel_ids

        with verification_scope(include_rejected=False):
            triples = adapter.get_event_centric_neighbors([ids["alice"]], BRAIN_ID)
        assert triples == []

        with verification_scope(include_rejected=True):
            triples = adapter.get_event_centric_neighbors([ids["alice"]], BRAIN_ID)
        assert len(triples) == 1
        assert triples[0][3].uuid == ids["targeted"]
        assert triples[0][3].verification_state == REJECTED
        assert triples[0][1].verification_state == VERIFIED

        traversed = adapter.traverse_graph(
            brain_id=BRAIN_ID,
            start_uuid=ids["alice"],
            depth=2,
            limit=20,
        )
        hop_rel_ids = {hop["predicate"]["uuid"] for hop in traversed["hops"]}
        assert ids["made"] in hop_rel_ids
        assert ids["targeted"] not in hop_rel_ids

        unverified_pred = Predicate(uuid="rel-live", name="LIVES_IN", description="lives")
        verified_candidate = {
            "triple": (
                Node(uuid="alice-1", name="Alice", labels=["PERSON"]),
                Predicate(
                    uuid=ids["made"],
                    name="MADE",
                    description="made",
                    properties={"verification_state": VERIFIED, "verified_by": "analyst-1"},
                ),
                Node(uuid="event-1", name="Meeting", labels=["EVENT"]),
                unverified_pred,
                Node(uuid="bob-1", name="Bob", labels=["PERSON"]),
            ),
            "score": 0.4,
            "ppr_mass": 0.0,
        }
        unverified_candidate = {
            "triple": (
                Node(uuid="alice-1", name="Alice", labels=["PERSON"]),
                Predicate(uuid="rel-other", name="MADE", description="made"),
                Node(uuid="event-2", name="Other", labels=["EVENT"]),
                unverified_pred,
                Node(uuid="carol-1", name="Carol", labels=["PERSON"]),
            ),
            "score": 0.4,
            "ppr_mass": 0.0,
        }
        ranked = retrieve_mod._rank_facts_with_completeness(
            [unverified_candidate, verified_candidate]
        )
        assert ranked[0]["triple"][1].uuid == ids["made"]

        got = api.get(f"/model/facts/{ids['made']}")
        assert got.status_code == 200
        assert got.json()["verification"]["verified_by"] == "analyst-1"

        reset = api.post(
            f"/model/facts/{ids['made']}/reset",
            json={"verified_by": "analyst-1", "note": "reopen"},
        )
        assert reset.status_code == 200, reset.text
        reset_body = reset.json()
        assert reset_body["verification"]["state"] == UNVERIFIED
        assert reset_body["verification"]["verified_by"] is None
        assert len(reset_body["history"]) == 2
        assert reset_body["history"][1]["to_state"] == UNVERIFIED

        missing = api.post(
            "/model/facts/does-not-exist/verify",
            json={"verified_by": "analyst-1"},
        )
        assert missing.status_code == 404


def test_new_facts_default_to_unverified():
    client = _memory_graph_client()
    adapter = _adapter_for(client)
    with _patched_graph_adapter(adapter):
        api = TestClient(_app())
        created = api.post(
            "/model/entity",
            json={"name": "Dana", "labels": ["PERSON"], "description": "person"},
        )
        assert created.status_code == 200, created.text
        other = api.post(
            "/model/entity",
            json={"name": "Orbit", "labels": ["ORG"], "description": "org"},
        )
        assert other.status_code == 200, other.text
        rel = api.post(
            "/model/relationship",
            json={
                "subject_uuid": created.json()[0]["uuid"],
                "object_uuid": other.json()[0]["uuid"],
                "predicate_name": "WORKS_AT",
                "predicate_description": "works",
            },
        )
        assert rel.status_code == 200, rel.text
        listed = api.get("/retrieve/relationships")
        assert listed.status_code == 200
        predicates = [
            item["predicate"] for item in listed.json()["relationships"]
        ]
        assert predicates
        assert all(
            pred.get("verification_state", UNVERIFIED) == UNVERIFIED
            for pred in predicates
        )
