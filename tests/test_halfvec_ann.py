import os
import unittest
from contextlib import contextmanager
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
}
for key, value in ENV_DEFAULTS.items():
    os.environ.setdefault(key, value)

from src.constants.embeddings import Vector
from src.lib.postgresql import vectors as vectors_mod
from src.utils.vector_search import stable_top_k_vectors


def _postgres_available() -> bool:
    try:
        import psycopg2

        from src.config import config

        pg = config.postgresql
        pg.validate_credentials()
        conn = psycopg2.connect(
            host=pg.host,
            port=pg.port,
            user=pg.username,
            password=pg.password,
            dbname=pg.maintenance_database or "postgres",
            connect_timeout=2,
        )
        conn.close()
        return True
    except Exception:
        return False


class HalfvecAnnTests(unittest.TestCase):
    def test_3072_creates_halfvec_hnsw_when_search_on(self):
        ddl = vectors_mod._vector_index_ddl(
            "vectors_data", 3072, search_enabled=True
        )
        self.assertIn("halfvec(3072)", ddl)
        self.assertIn("hnsw", ddl)
        self.assertIn("halfvec_cosine_ops", ddl)

    def test_3072_skips_index_when_search_off(self):
        ddl = vectors_mod._vector_index_ddl(
            "vectors_data", 3072, search_enabled=False
        )
        self.assertEqual(ddl.strip(), "")

    def test_under_2000_keeps_float32_hnsw(self):
        ddl = vectors_mod._vector_index_ddl(
            "vectors_data", 1536, search_enabled=True
        )
        self.assertIn("vector_cosine_ops", ddl)
        self.assertNotIn("halfvec", ddl)

    def test_overfetch_float32_rerank_matches_exact_topk(self):
        ann_order = [
            Vector(id="b", metadata={"uuid": "b", "resource_id": "b"}, distance=0.05),
            Vector(id="a", metadata={"uuid": "a", "resource_id": "a"}, distance=0.01),
            Vector(id="c", metadata={"uuid": "c", "resource_id": "c"}, distance=0.20),
        ]
        reranked = stable_top_k_vectors(ann_order, 2)
        self.assertEqual([v.id for v in reranked], ["a", "b"])

    def test_search_vectors_orders_by_halfvec_and_reranks_float32(self):
        client = vectors_mod.PostgreSQLVectorStoreClient.__new__(
            vectors_mod.PostgreSQLVectorStoreClient
        )
        executed: list[str] = []

        class Cursor:
            def execute(self, sql, params=None):
                executed.append(str(sql))

            def fetchall(self):
                return [
                    {
                        "id": 2,
                        "uuid": "b",
                        "metadata": {"resource_id": "b"},
                        "distance": 0.2,
                    },
                    {
                        "id": 1,
                        "uuid": "a",
                        "metadata": {"resource_id": "a"},
                        "distance": 0.05,
                    },
                ]

        @contextmanager
        def fake_connection(_brain_id):
            conn = MagicMock()
            cur = Cursor()

            @contextmanager
            def fake_cursor(**_kwargs):
                yield cur

            conn.cursor.side_effect = lambda **kwargs: fake_cursor()
            yield conn

        with (
            patch.object(client, "_ensure_store"),
            patch.object(client, "_connection", fake_connection),
            patch.object(vectors_mod, "EMBEDDING_STORES_SIZES", {"data": 3072}),
            patch.object(vectors_mod.config, "search_enabled", True),
        ):
            hits = client.search_vectors([0.0] * 3072, "brain", "data", k=1)

        joined = "\n".join(executed)
        self.assertIn("halfvec(3072)", joined)
        self.assertIn("embeddings <=> %s::vector", joined)
        self.assertEqual([h.id for h in hits], ["a"])

    @unittest.skipUnless(_postgres_available(), "postgres not available")
    def test_postgres_creates_halfvec_hnsw_for_3072(self):
        import psycopg2.extras

        from src.lib.postgresql._naming import brain_db_name
        from src.lib.postgresql._provisioning import borrow_brain

        brain_id = "searchbench_halfvec_unit"
        sizes = {
            "nodes": 3072,
            "triplets": 3072,
            "observations": 3072,
            "data": 3072,
            "relationships": 3072,
        }
        client = vectors_mod.PostgreSQLVectorStoreClient()
        with (
            patch.object(vectors_mod.config, "search_enabled", True),
            patch.object(vectors_mod, "EMBEDDING_STORES_SIZES", sizes),
        ):
            client._initialized_stores.clear()
            client._ensure_store("data", brain_id)
        with borrow_brain(brain_id) as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT indexdef
                    FROM pg_indexes
                    WHERE tablename = 'vectors_data'
                      AND indexname = 'idx_vectors_data_embeddings_halfvec'
                    """
                )
                row = cur.fetchone()
        self.assertIsNotNone(row)
        self.assertIn("halfvec", row["indexdef"])
        self.assertIn("hnsw", row["indexdef"].lower())


if __name__ == "__main__":
    unittest.main()
