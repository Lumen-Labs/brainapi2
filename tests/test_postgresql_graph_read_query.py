import json
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

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
    "POSTGRES_PASSWORD": "postgres",
}
for key, value in ENV_DEFAULTS.items():
    os.environ.setdefault(key, value)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import importlib.util
from pathlib import Path

_read_query_path = (
    Path(__file__).resolve().parent.parent / "src/lib/postgresql/read_query.py"
)
_read_query_spec = importlib.util.spec_from_file_location(
    "brainapi_read_query", _read_query_path
)
_read_query_module = importlib.util.module_from_spec(_read_query_spec)
assert _read_query_spec.loader is not None
_read_query_spec.loader.exec_module(_read_query_module)
validate_read_only_sql = _read_query_module.validate_read_only_sql
ReadQueryValidationError = _read_query_module.ReadQueryValidationError

from src.adapters.graph import GraphAdapter

try:
    from src.lib.postgresql.graph_store import GraphDatabaseError, PostgreSQLGraphStore
except ImportError:
    GraphDatabaseError = Exception
    PostgreSQLGraphStore = None


class ValidateReadOnlySqlTests(unittest.TestCase):
    def test_accepts_select(self):
        sql = validate_read_only_sql("SELECT uuid, data FROM kg_nodes LIMIT 5")
        self.assertEqual(sql, "SELECT uuid, data FROM kg_nodes LIMIT 5")

    def test_accepts_with_recursive(self):
        query = (
            "WITH RECURSIVE walk AS ("
            "SELECT source_uuid, target_uuid FROM kg_relationships "
            "WHERE source_uuid = 'abc' "
            "UNION ALL SELECT r.source_uuid, r.target_uuid "
            "FROM walk w JOIN kg_relationships r ON r.source_uuid = w.target_uuid "
            "WHERE w.depth < 2) SELECT * FROM walk"
        )
        self.assertTrue(validate_read_only_sql(query).startswith("WITH RECURSIVE"))

    def test_accepts_table_shorthand(self):
        sql = validate_read_only_sql("TABLE kg_nodes")
        self.assertEqual(sql, "TABLE kg_nodes")

    def test_rejects_empty_query(self):
        with self.assertRaises(ReadQueryValidationError):
            validate_read_only_sql("   ")

    def test_rejects_multiple_statements(self):
        with self.assertRaises(ReadQueryValidationError):
            validate_read_only_sql("SELECT 1; DELETE FROM kg_nodes")

    def test_rejects_insert(self):
        with self.assertRaises(ReadQueryValidationError):
            validate_read_only_sql("INSERT INTO kg_nodes VALUES ('x', '{}')")

    def test_rejects_delete(self):
        with self.assertRaises(ReadQueryValidationError):
            validate_read_only_sql("DELETE FROM kg_nodes")

    def test_rejects_drop(self):
        with self.assertRaises(ReadQueryValidationError):
            validate_read_only_sql("DROP TABLE kg_nodes")

    def test_rejects_invalid_start_keyword(self):
        with self.assertRaises(ReadQueryValidationError):
            validate_read_only_sql("UPDATE kg_nodes SET data = '{}'")

    def test_rejects_cypher_match(self):
        with self.assertRaises(ReadQueryValidationError) as ctx:
            validate_read_only_sql(
                "MATCH (n:PERSON) WHERE n.uuid = 'abc' RETURN n"
            )
        self.assertIn("Cypher", str(ctx.exception))

    def test_rejects_explain_cypher(self):
        with self.assertRaises(ReadQueryValidationError) as ctx:
            validate_read_only_sql(
                "EXPLAIN MATCH (n) RETURN n.uuid AS uuid LIMIT 50"
            )
        self.assertIn("Cypher", str(ctx.exception))

    def test_rejects_legacy_nodes_table(self):
        with self.assertRaises(ReadQueryValidationError) as ctx:
            validate_read_only_sql("SELECT COUNT(*) AS total FROM nodes")
        self.assertIn("kg_nodes", str(ctx.exception))

    def test_rejects_table_nodes_shorthand(self):
        with self.assertRaises(ReadQueryValidationError) as ctx:
            validate_read_only_sql("TABLE nodes")
        self.assertIn("kg_nodes", str(ctx.exception))

    def test_rejects_incomplete_with_recursive_unbalanced_parens(self):
        with self.assertRaises(ReadQueryValidationError) as ctx:
            validate_read_only_sql(
                "WITH RECURSIVE walk AS ("
                "SELECT source_uuid, target_uuid, rel_type, 1 AS depth "
                "FROM kg_relationships WHERE source_uuid = 'abc'"
            )
        self.assertIn("Incomplete SQL", str(ctx.exception))

    def test_rejects_recursive_cte_fragment_without_outer_select(self):
        with self.assertRaises(ReadQueryValidationError) as ctx:
            validate_read_only_sql(
                "WITH RECURSIVE walk AS ("
                "SELECT source_uuid, target_uuid, rel_type, 1 AS depth "
                "FROM kg_relationships WHERE source_uuid = 'abc')"
            )
        msg = str(ctx.exception)
        self.assertTrue(
            "Incomplete WITH RECURSIVE" in msg or "Incomplete SQL" in msg
        )

    def test_rejects_orphaned_union_all_branch(self):
        with self.assertRaises(ReadQueryValidationError) as ctx:
            validate_read_only_sql(
                "SELECT r.source_uuid, r.target_uuid, r.rel_type, w.depth + 1 "
                "FROM walk w JOIN kg_relationships r ON r.source_uuid = w.target_uuid "
                "WHERE w.depth < 3) SELECT DISTINCT w.source_uuid FROM walk w LIMIT 50"
            )
        self.assertIn("Incomplete SQL", str(ctx.exception))


class ExecuteReadQueryTests(unittest.TestCase):
    @unittest.skipIf(PostgreSQLGraphStore is None, "postgresql dependencies unavailable")
    def test_applies_row_cap_and_truncated_flag(self):
        store = PostgreSQLGraphStore()
        rows = [{"uuid": f"n{i}", "data": {"name": f"node{i}"}} for i in range(101)]
        mock_cursor = MagicMock()
        mock_cursor.fetchmany.return_value = rows
        mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
        mock_cursor.__exit__ = MagicMock(return_value=False)

        mock_conn = MagicMock()
        mock_conn.autocommit = True
        mock_conn.cursor.return_value = mock_cursor
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)

        with patch.object(store, "_connection", return_value=mock_conn):
            with patch.object(store, "_ensure_brain_schema"):
                result = store.execute_read_query("default", "SELECT uuid, data FROM kg_nodes")

        executed = [call.args[0] for call in mock_cursor.execute.call_args_list]
        self.assertTrue(any("SET TRANSACTION READ ONLY" in str(sql) for sql in executed))
        self.assertTrue(result["truncated"])
        self.assertEqual(len(result["records"]), 100)


class GraphOperationSerializationTests(unittest.TestCase):
    def test_execute_operation_serializes_postgresql_read_result(self):
        class FakeGraphClient:
            def execute_operation(self, operation: str, brain_id: str):
                return {
                    "records": [{"uuid": "abc", "data": {"name": "Alice"}}],
                    "truncated": False,
                }

        adapter = GraphAdapter()
        adapter.add_client(FakeGraphClient())
        result = adapter.execute_operation(
            "SELECT uuid, data FROM kg_nodes LIMIT 1",
            brain_id="test-brain",
        )
        payload = json.loads(result)
        self.assertFalse(payload["truncated"])
        self.assertEqual(payload["records"][0]["uuid"], "abc")

    def test_execute_operation_returns_error_string_on_validation_failure(self):
        class FakeGraphClient:
            def execute_operation(self, operation: str, brain_id: str):
                raise GraphDatabaseError("Only read-only SELECT queries are allowed")

        adapter = GraphAdapter()
        adapter.add_client(FakeGraphClient())
        result = adapter.execute_operation("DELETE FROM kg_nodes", brain_id="default")
        self.assertIn("Error executing graph operation", result)
        self.assertIn("read-only", result)


if __name__ == "__main__":
    unittest.main()
