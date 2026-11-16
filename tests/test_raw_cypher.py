"""Unit tests for read-only raw Cypher execution."""

from __future__ import annotations

import os
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

import pytest
from neo4j import Query, READ_ACCESS, WRITE_ACCESS

from src.adapters.graph import GraphAdapter
from src.adapters.graph_operation_result_serializer import (
    serialize_graph_operation_result,
)
from src.config import config
from src.lib.cypher_safety import (
    MAX_RAW_QUERY_ROWS,
    RAW_QUERY_TIMEOUT_SECONDS,
    InvalidCypherIdentifier,
    RawCypherRejected,
)
from src.lib.neo4j.client import Neo4jClient
from src.lib.postgresql.read_query import ReadQueryValidationError, validate_read_only_sql


class _IterResult:
    def __init__(self, records, keys=("n",)):
        self._records = records
        self._keys = keys

    def keys(self):
        return list(self._keys)

    def __iter__(self):
        return iter(self._records)


class _Session:
    def __init__(self, driver, database, access_mode):
        self.driver = driver
        self.database = database
        self.access_mode = access_mode

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def run(self, query):
        self.driver.runs.append(query)
        return _IterResult(self.driver.records, self.driver.keys)


class _Driver:
    def __init__(self, records=None, keys=("n",)):
        self.records = records or []
        self.keys = keys
        self.runs = []
        self.session_calls = []

    def session(self, database=None, default_access_mode=None):
        self.session_calls.append((database, default_access_mode))
        return _Session(self, database, default_access_mode)

    def execute_query(self, *args, **kwargs):
        raise AssertionError("raw execute_operation must not use execute_query")


def _client(records=None, keys=("n",)) -> Neo4jClient:
    client = Neo4jClient.__new__(Neo4jClient)
    client.driver = _Driver(records=records, keys=keys)
    client.ensure_database = MagicMock()
    return client


@pytest.fixture(autouse=True)
def _restore_write_flag():
    original = config.graph_raw_write_enabled
    config.graph_raw_write_enabled = False
    try:
        yield
    finally:
        config.graph_raw_write_enabled = original


def test_execute_operation_uses_read_session_timeout_and_brain_database():
    client = _client(records=[{"n": 1}])
    result = client.execute_operation("RETURN 1 AS n", "braina")
    assert client.driver.session_calls == [("braina", READ_ACCESS)]
    assert len(client.driver.runs) == 1
    query = client.driver.runs[0]
    assert isinstance(query, Query)
    assert query.text == "RETURN 1 AS n"
    assert query.timeout == RAW_QUERY_TIMEOUT_SECONDS
    assert result.records == [{"n": 1}]
    assert result.keys() == ["n"]
    assert result.truncated is False
    client.ensure_database.assert_not_called()


def test_execute_operation_uses_write_session_only_when_flag_enabled():
    client = _client(records=[])
    config.graph_raw_write_enabled = True
    client.execute_operation("CREATE (n:Person {name: 'Ada'}) RETURN n", "braina")
    assert client.driver.session_calls == [("braina", WRITE_ACCESS)]


def test_execute_operation_rejects_cross_database_and_admin_without_session():
    client = _client()
    rejected = [
        "USE system MATCH (n) RETURN n",
        "USE neo4j MATCH (n) RETURN n",
        "CREATE DATABASE other",
        "DROP DATABASE neo4j",
        "SHOW DATABASES",
        "CALL dbms.components() YIELD name RETURN name",
        "CALL dbms.security.listUsers() YIELD user RETURN user",
        "RETURN 1; MATCH (n) DETACH DELETE n",
    ]
    for query in rejected:
        with pytest.raises(RawCypherRejected):
            client.execute_operation(query, "braina")
    assert client.driver.session_calls == []
    assert client.driver.runs == []


def test_execute_operation_rejects_database_admin_even_when_writes_enabled():
    client = _client()
    config.graph_raw_write_enabled = True
    with pytest.raises(RawCypherRejected):
        client.execute_operation("CREATE DATABASE stolen", "braina")
    with pytest.raises(RawCypherRejected):
        client.execute_operation("USE system RETURN 1", "braina")
    assert client.driver.session_calls == []


def test_execute_operation_rejects_injected_and_system_database_names():
    client = _client()
    with pytest.raises(InvalidCypherIdentifier):
        client.execute_operation("RETURN 1", "neo4j`; MATCH (n) DETACH DELETE n")
    with pytest.raises(InvalidCypherIdentifier):
        client.execute_operation("RETURN 1", "system")
    assert client.driver.session_calls == []


def test_execute_operation_caps_result_rows():
    records = [{"n": i} for i in range(MAX_RAW_QUERY_ROWS + 25)]
    client = _client(records=records)
    result = client.execute_operation("UNWIND range(1, 200) AS n RETURN n", "braina")
    assert len(result.records) == MAX_RAW_QUERY_ROWS
    assert result.truncated is True


def test_adapter_returns_error_string_for_rejected_raw_cypher():
    adapter = GraphAdapter()
    adapter.add_client(_client())
    result = adapter.execute_operation("USE system MATCH (n) RETURN n", brain_id="braina")
    assert result.startswith("Error executing graph operation:")
    assert "switch databases" in result


def test_postgres_raw_sql_stays_read_only_when_graph_raw_write_enabled():
    config.graph_raw_write_enabled = True
    with pytest.raises(ReadQueryValidationError):
        validate_read_only_sql("DELETE FROM kg_nodes")
    with pytest.raises(ReadQueryValidationError):
        validate_read_only_sql("DROP TABLE kg_relationships")


def test_serializer_honors_execution_truncated_flag():
    class _Result:
        records = [{"n": 1}]
        truncated = True

        def keys(self):
            return ["n"]

    payload = serialize_graph_operation_result(_Result())
    assert '"truncated": true' in payload
