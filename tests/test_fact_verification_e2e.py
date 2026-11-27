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
from src.constants.data import TextChunk
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
def _patched_graph_adapter(adapter: GraphAdapter, data=None):
    data = data if data is not None else _MemoryChunks()
    patches = [
        patch("src.services.api.controllers.model.graph_adapter", adapter),
        patch("src.services.api.controllers.retrieve.graph_adapter", adapter),
        patch("src.services.api.controllers.verification.graph_adapter", adapter),
        patch("src.core.search.entities.graph_adapter", adapter),
        patch("src.core.search.relationships.graph_adapter", adapter),
        patch("src.services.kg_agent.main.graph_adapter", adapter),
        patch("src.services.data.main.data_adapter", data),
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
        request.state.principal = f"brainpat:{brain_id}"
        request.state.scopes = ["brainapi", "verification"]
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
    tagged = Predicate(
        uuid="rel-tagged",
        name="MENTIONED",
        description="mentioned",
        flow_key="flow-2",
        properties={
            "interview_id": "int-9",
            "topic_id": "top-3",
            "source_key": "src-notes",
            "source_chunk_id": "chunk-77",
        },
    )
    adapter.add_relationship(alice, tagged, bob, BRAIN_ID)
    return {
        "alice": alice.uuid,
        "event": event.uuid,
        "bob": bob.uuid,
        "made": made.uuid,
        "targeted": targeted.uuid,
        "tagged": tagged.uuid,
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
        assert verified_body["verification"]["verified_by"] == f"brainpat:{BRAIN_ID}"
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

        rejected_event = api.post(
            f"/model/facts/{ids['event']}/reject",
            json={"verified_by": "analyst-1", "note": "bad event"},
        )
        assert rejected_event.status_code == 200, rejected_event.text
        after_event = api.get("/retrieve/relationships")
        assert after_event.status_code == 200
        after_ids = {
            (item.get("predicate") or {}).get("uuid")
            for item in after_event.json().get("relationships") or []
        }
        assert ids["made"] not in after_ids
        assert ids["targeted"] not in after_ids
        after_traverse = adapter.traverse_graph(
            brain_id=BRAIN_ID,
            start_uuid=ids["alice"],
            depth=2,
            limit=20,
        )
        after_hop_ids = {hop["predicate"]["uuid"] for hop in after_traverse["hops"]}
        assert ids["made"] not in after_hop_ids
        assert ids["targeted"] not in after_hop_ids

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
        assert got.json()["verification"]["verified_by"] == f"brainpat:{BRAIN_ID}"

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

        person_reject = api.post(
            f"/model/facts/{ids['alice']}/reject",
            json={"note": "should not hide a person node"},
        )
        assert person_reject.status_code == 404
        alice_row = adapter.get_fact_record(ids["alice"], BRAIN_ID)
        assert alice_row is None
        listed_after = api.get("/model/facts", params={"include_rejected": True})
        listed_after_ids = {item["fact_id"] for item in listed_after.json()["facts"]}
        assert ids["alice"] not in listed_after_ids

        event_reject = api.post(
            f"/model/facts/{ids['event']}/reject",
            json={"note": "event nodes are verifiable facts"},
        )
        assert event_reject.status_code == 200, event_reject.text
        assert event_reject.json()["verification"]["state"] == REJECTED
        listed_rejected = api.get(
            "/model/facts", params={"state": "rejected", "include_rejected": True}
        )
        listed_rejected_ids = {
            item["fact_id"] for item in listed_rejected.json()["facts"]
        }
        assert ids["event"] in listed_rejected_ids

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


def _app_without_verification(brain_id: str = BRAIN_ID) -> FastAPI:
    app = FastAPI()
    install_error_handlers(app)

    @app.middleware("http")
    async def inject_brain(request: Request, call_next):
        request.state.brain_id = brain_id
        request.state.principal = f"brainpat:{brain_id}"
        request.state.scopes = ["brainapi"]
        return await call_next(request)

    app.include_router(model_router)
    app.include_router(retrieve_router)
    return app


def test_verify_rejects_plain_brain_token():
    from src.core.auth.scopes import VERIFICATION_PERMISSION_DENIED

    client = _memory_graph_client()
    adapter = _adapter_for(client)
    ids = _seed_facts(adapter)
    with _patched_graph_adapter(adapter):
        api = TestClient(_app_without_verification())
        listed = api.get("/model/facts")
        assert listed.status_code == 200
        denied = api.post(
            f"/model/facts/{ids['made']}/verify",
            json={"verification_source": "doc-1"},
        )
        assert denied.status_code == 403
        body = denied.json()
        assert body["error"]["code"] == VERIFICATION_PERMISSION_DENIED
        row = adapter.get_fact_record(ids["made"], BRAIN_ID)
        assert (row.get("properties") or {}).get("verification_state", UNVERIFIED) == UNVERIFIED


def test_concurrent_verification_appends_history():
    from src.core.verification.service import apply_verification_action

    client = _memory_graph_client()
    adapter = _adapter_for(client)
    ids = _seed_facts(adapter)
    barrier = threading.Barrier(2)
    errors = []

    def _reject(note):
        try:
            barrier.wait()
            apply_verification_action(
                adapter,
                fact_id=ids["made"],
                brain_id=BRAIN_ID,
                action="reject",
                actor=f"brainpat:{note}",
                note=note,
            )
        except Exception as exc:
            errors.append(exc)

    threads = [
        threading.Thread(target=_reject, args=("one",)),
        threading.Thread(target=_reject, args=("two",)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not errors
    row = adapter.get_fact_record(ids["made"], BRAIN_ID)
    from src.core.verification.facts import parse_history

    history = parse_history((row.get("properties") or {}).get("verification_history"))
    assert len(history) == 2
    assert {entry.note for entry in history} == {"one", "two"}


def test_list_facts_returns_source_metadata_and_filters_server_side():
    client = _memory_graph_client()
    adapter = _adapter_for(client)
    ids = _seed_facts(adapter)
    with _patched_graph_adapter(adapter):
        api = TestClient(_app())

        listed = api.get("/model/facts")
        assert listed.status_code == 200, listed.text
        body = listed.json()
        unfiltered_total = body["total"]
        by_id = {item["fact_id"]: item for item in body["facts"]}
        tagged = by_id[ids["tagged"]]
        assert tagged["interview_id"] == "int-9"
        assert tagged["topic_id"] == "top-3"
        assert tagged["source_key"] == "src-notes"
        assert tagged["source_chunk_id"] == "chunk-77"
        assert tagged["source"] == "src-notes"
        assert tagged["meta_keys"]["interviewId"] == "int-9"
        assert tagged["meta_keys"]["topicId"] == "top-3"
        assert tagged["meta_keys"]["source"] == "src-notes"
        assert by_id[ids["made"]].get("interview_id") in (None, "")

        by_interview = api.get("/model/facts", params={"interview_id": "int-9"})
        assert by_interview.status_code == 200, by_interview.text
        interview_body = by_interview.json()
        assert interview_body["total"] == 1
        assert interview_body["filteredTotal"] == 1
        assert interview_body["facts"][0]["fact_id"] == ids["tagged"]
        assert interview_body["total"] < unfiltered_total

        by_topic = api.get("/model/facts", params={"topic_id": "top-3"})
        assert {item["fact_id"] for item in by_topic.json()["facts"]} == {ids["tagged"]}
        assert by_topic.json()["total"] == 1

        by_source = api.get("/model/facts", params={"source": "src-notes"})
        assert {item["fact_id"] for item in by_source.json()["facts"]} == {ids["tagged"]}
        assert by_source.json()["total"] == 1

        by_entity = api.get("/model/facts", params={"entity_id": ids["alice"]})
        entity_ids = {item["fact_id"] for item in by_entity.json()["facts"]}
        assert ids["tagged"] in entity_ids
        assert ids["made"] in entity_ids
        assert ids["targeted"] not in entity_ids
        assert by_entity.json()["total"] == len(entity_ids)

        paged = api.get(
            "/model/facts",
            params={"entity_id": ids["alice"], "limit": 1, "skip": 0},
        )
        assert paged.status_code == 200
        assert paged.json()["total"] == by_entity.json()["total"]
        assert len(paged.json()["facts"]) == 1


class _MemoryChunks:
    def __init__(self):
        self.chunks = {}

    def save_text_chunk(self, chunk, brain_id="default"):
        self.chunks[chunk.id] = chunk
        return chunk

    def get_text_chunks_by_ids(self, ids, with_observations, brain_id):
        found = [self.chunks[item] for item in ids if item in self.chunks]
        return found, []

    def get_text_chunks(
        self,
        brain_id="default",
        limit=10,
        skip=0,
        query_text=None,
        metadata_eq=None,
        order="desc",
    ):
        items = list(self.chunks.values())
        if metadata_eq:
            matched = []
            for chunk in items:
                meta = chunk.metadata or {}
                if all(str(meta.get(key)) == str(value) for key, value in metadata_eq.items()):
                    matched.append(chunk)
            items = matched
        total = len(items)
        return items[skip : skip + limit], total


def test_list_facts_resolves_interview_id_through_ingest_chunks():
    client = _memory_graph_client()
    adapter = _adapter_for(client)
    data = _MemoryChunks()
    alice = adapter.add_nodes(
        [Node(uuid="alice-i", name="Alice", labels=["PERSON"], description="person")],
        BRAIN_ID,
    )[0]
    bob = adapter.add_nodes(
        [Node(uuid="bob-i", name="Bob", labels=["PERSON"], description="person")],
        BRAIN_ID,
    )[0]
    wanted = data.save_text_chunk(
        TextChunk(
            id="chunk-interview-a",
            text="Alice knows Bob from the interview.",
            metadata={"interview_id": "int-ingest", "topicId": "top-ingest", "source": "notes"},
        ),
        BRAIN_ID,
    )
    other = data.save_text_chunk(
        TextChunk(
            id="chunk-interview-b",
            text="Other import about Carol.",
            metadata={"interview_id": "int-other"},
        ),
        BRAIN_ID,
    )
    kept = Predicate(
        uuid="rel-interview-a",
        name="KNOWS",
        description="knows",
        properties={"source_chunk_ids": [wanted.id]},
    )
    skipped = Predicate(
        uuid="rel-interview-b",
        name="MENTIONED",
        description="mentioned",
        properties={"source_chunk_ids": [other.id]},
    )
    adapter.add_relationship(alice, kept, bob, BRAIN_ID)
    adapter.add_relationship(alice, skipped, bob, BRAIN_ID)
    with _patched_graph_adapter(adapter, data=data):
        api = TestClient(_app())
        listed = api.get("/model/facts", params={"interview_id": "int-ingest"})
        assert listed.status_code == 200, listed.text
        body = listed.json()
        fact_ids = {item["fact_id"] for item in body["facts"]}
        assert kept.uuid in fact_ids
        assert skipped.uuid not in fact_ids
        assert body["total"] == 1
        assert body["filteredTotal"] == 1
        fact = body["facts"][0]
        assert fact["interview_id"] == "int-ingest"
        assert fact["topic_id"] == "top-ingest"
        assert fact["source"] == "notes"
        assert fact["meta_keys"]["interviewId"] == "int-ingest"
        assert fact["meta_keys"]["topicId"] == "top-ingest"
        assert fact["meta_keys"]["source"] == "notes"

        other_list = api.get("/model/facts", params={"interview_id": "int-other"})
        assert {item["fact_id"] for item in other_list.json()["facts"]} == {skipped.uuid}
