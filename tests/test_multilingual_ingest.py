import os
import threading
from unittest.mock import MagicMock

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

from src.constants.kg import Node
from src.lib.cypher_safety import InvalidCypherIdentifier, clean_label
from src.lib.postgresql.graph_store import PostgreSQLGraphStore, _BrainGraph
from src.lib.postgresql.networkx_client import NetworkXGraphClient


BRAIN_ID = "multilingual-ingest"


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


def test_multilingual_entity_types_ingest_without_failing_the_batch():
    client = _memory_graph_client()
    added = client.add_nodes(
        [
            Node(uuid="city-1", name="Roma", labels=["CITTÀ"], properties={}),
            Node(uuid="name-1", name="Hanako", labels=["名前"], properties={}),
            Node(uuid="person-1", name="Ada", labels=["PERSON"], properties={}),
            Node(uuid="bad-1", name="Skip me", labels=["PERSON`DETACH DELETE"], properties={}),
        ],
        BRAIN_ID,
    )
    assert [node.uuid for node in added] == ["city-1", "name-1", "person-1"]
    brain = client._store.get_brain(BRAIN_ID)
    assert set(brain.labels("city-1")) == {"CITTA"}
    assert brain.node_data("city-1")["original_type"] == "CITTÀ"
    assert set(brain.labels("name-1")) == {clean_label("名前")}
    assert brain.node_data("name-1")["original_type"] == "名前"
    assert set(brain.labels("person-1")) == {"PERSON"}
    assert "bad-1" not in brain.graph.nodes


def test_single_invalid_label_still_raises_for_api_style_calls():
    client = _memory_graph_client()
    try:
        client.add_nodes(
            [Node(uuid="only-bad", name="Skip", labels=["PERSON`x"], properties={})],
            BRAIN_ID,
        )
        raised = False
    except InvalidCypherIdentifier:
        raised = True
    assert raised
    brain = client._store.get_brain(BRAIN_ID)
    assert list(brain.graph.nodes) == []
