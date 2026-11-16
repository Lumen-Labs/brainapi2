from __future__ import annotations

import os
import threading
from unittest.mock import MagicMock, Mock

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
from src.core.saving.ingest_publication import replay_graph_operation
from src.core.saving.ingest_write_buffer import capture_writes
from src.core.verification.facts import UNVERIFIED, VERIFIED, extract_verification
from src.core.verification.service import apply_verification_action
from src.lib.postgresql.graph_store import PostgreSQLGraphStore, _BrainGraph
from src.lib.postgresql.networkx_client import NetworkXGraphClient

BRAIN_ID = "stagedverif"


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


def test_set_fact_verification_does_not_touch_live_store_until_replay():
    graph = GraphAdapter()
    graph.graph = Mock()
    properties = {
        "verification_state": VERIFIED,
        "verified_by": "analyst-1",
        "verified_at": "2026-11-16T00:00:00+00:00",
        "verification_source": "doc-1",
        "verification_note": "ok",
        "verification_history": ["{}"],
    }
    with capture_writes("brain") as operations:
        graph.set_fact_verification(
            "rel-1",
            "brain",
            kind="relationship",
            properties=properties,
        )
    graph.graph.set_fact_verification.assert_not_called()
    assert operations[0]["method"] == "set_fact_verification"
    assert operations[0]["args"]["fact_id"] == "rel-1"
    assert operations[0]["args"]["kind"] == "relationship"
    replay_graph_operation(graph.graph, "brain", operations[0])
    graph.graph.set_fact_verification.assert_called_once_with(
        "rel-1",
        "brain",
        kind="relationship",
        properties=properties,
    )


def test_staged_verification_replays_onto_memory_graph():
    client = _memory_graph_client()
    adapter = GraphAdapter()
    adapter.add_client(client)
    alice = adapter.add_nodes(
        [Node(uuid="alice-1", name="Alice", labels=["PERSON"])],
        BRAIN_ID,
    )[0]
    acme = adapter.add_nodes(
        [Node(uuid="acme-1", name="Acme", labels=["ORG"])],
        BRAIN_ID,
    )[0]
    predicate = Predicate(uuid="rel-1", name="WORKS_AT", description="works")
    adapter.add_relationship(alice, predicate, acme, BRAIN_ID)

    with capture_writes(BRAIN_ID) as operations:
        staged = apply_verification_action(
            adapter,
            fact_id="rel-1",
            brain_id=BRAIN_ID,
            action="verify",
            verified_by="analyst-1",
            verification_source="doc-1",
        )
    assert staged.verification.state == VERIFIED
    live = adapter.get_fact_record("rel-1", BRAIN_ID)
    assert extract_verification(live).state == UNVERIFIED
    assert operations and operations[0]["method"] == "set_fact_verification"

    replay_graph_operation(adapter.graph, BRAIN_ID, operations[0])
    published = adapter.get_fact_record("rel-1", BRAIN_ID)
    assert extract_verification(published).state == VERIFIED
    assert extract_verification(published).verified_by == "analyst-1"
