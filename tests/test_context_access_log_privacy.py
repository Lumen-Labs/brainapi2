import io
import logging
import os
import sys
import unittest
from types import SimpleNamespace
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

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.constants.data import TextChunk
from src.constants.embeddings import Vector
from src.constants.kg import Node, Predicate
from src.core.saving.history_intent import reset_prototype_cache
from src.services.api.dependencies import get_brain_id
from src.services.api.errors import install_error_handlers
from src.services.api.logging_privacy import (
    AccessLogPrivacyFilter,
    install_access_log_privacy,
    redact_access_target,
)
from src.services.api.routes.retrieve import retrieve_router
from src.utils.nlp.ner import ExtractElementsResponse

PROBE_QUESTION = "PROBE_QUESTION_ABOUT_ALICE_NEIGHBOR"


def _keyword_chunk():
    return TextChunk(id="kw-1", text="Alice is mentioned in keyword search")


def _client():
    app = FastAPI()
    install_error_handlers(app)
    app.include_router(retrieve_router)
    app.dependency_overrides[get_brain_id] = lambda: "brain-a"
    return TestClient(app, raise_server_exceptions=True)


class ContextAccessLogPrivacyTests(unittest.TestCase):
    def tearDown(self):
        reset_prototype_cache()

    def test_get_context_prefers_post_in_openapi_description(self):
        app = FastAPI()
        app.include_router(retrieve_router)
        schema = app.openapi()
        description = schema["paths"]["/retrieve/context"]["get"].get("description", "")
        self.assertIn("Prefer POST", description)
        self.assertIn("sensitive", description.lower())

    def test_access_log_line_omits_get_context_query_text(self):
        from src.services.api.controllers import retrieve as retrieve_mod

        install_access_log_privacy()
        chunk = _keyword_chunk()
        client = _client()
        mock_embeddings = MagicMock()
        mock_embeddings.embed_text.return_value = Vector(
            id="q", embeddings=[0.1, 0.2, 0.3], metadata={}
        )
        mock_embeddings.embed_texts.side_effect = lambda texts: [
            Vector(id=f"b{i}", embeddings=[0.1, 0.2, 0.3], metadata={})
            for i, _ in enumerate(texts)
        ]
        mock_vs = MagicMock()
        mock_vs.search_nodes.return_value = [
            Vector(
                id="n1",
                embeddings=[0.1, 0.2, 0.3],
                metadata={"uuid": "a", "name": "Alice"},
                distance=0.1,
            )
        ]
        mock_vs.search_relationships.return_value = []
        mock_vs.search_data.return_value = []
        mock_graph = MagicMock()
        mock_graph.get_event_centric_neighbors.return_value = [
            (
                Node(uuid="a", name="Alice", labels=["PERSON"], description="d"),
                Predicate(uuid="r1", name="ATTENDED", description="d"),
                Node(
                    uuid="e1",
                    name="Event1",
                    labels=["EVENT"],
                    description="d",
                    happened_at="2024-01-01",
                ),
                Predicate(uuid="r2", name="WITH", description="d"),
                Node(uuid="b1", name="Person1", labels=["PERSON"], description="d"),
            )
        ]
        mock_graph.get_hub_bridges.return_value = []
        mock_graph.list_topic_memberships.return_value = []
        mock_data = MagicMock()
        mock_data.get_last_text_chunks.return_value = []
        mock_data.get_last_structured_data.return_value = []
        mock_data.search.return_value = SimpleNamespace(text_chunks=[chunk])
        mock_data.get_text_chunks_by_ids.return_value = ([chunk], [])
        with (
            patch.object(retrieve_mod, "embeddings_adapter", mock_embeddings),
            patch.object(retrieve_mod, "vector_search", mock_vs),
            patch.object(retrieve_mod, "graph_adapter", mock_graph),
            patch.object(retrieve_mod, "data_adapter", mock_data),
            patch.object(
                retrieve_mod._entity_extractor,
                "extract_elements",
                return_value=ExtractElementsResponse(
                    tokens=[{"text": "Alice", "lemma": "Alice", "pos": "PROPN"}],
                    noun_chunks=[],
                ),
            ),
        ):
            response = client.get(
                "/retrieve/context",
                params={"text": PROBE_QUESTION},
            )
        self.assertEqual(response.status_code, 200)
        query = response.request.url.query
        if isinstance(query, (bytes, bytearray)):
            query = query.decode()
        raw_target = response.request.url.path
        if query:
            raw_target = f"{raw_target}?{query}"
        self.assertIn(PROBE_QUESTION, raw_target)

        logger = logging.getLogger("uvicorn.access")
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
        previous_level = logger.level
        previous_propagate = logger.propagate
        logger.setLevel(logging.INFO)
        logger.propagate = False
        try:
            logger.info(
                '%s - "%s %s HTTP/%s" %d',
                "127.0.0.1:9",
                "GET",
                raw_target,
                "1.1",
                response.status_code,
            )
            line = stream.getvalue()
        finally:
            logger.removeHandler(handler)
            logger.setLevel(previous_level)
            logger.propagate = previous_propagate

        self.assertIn("/retrieve/context", line)
        self.assertIn("text=[redacted]", line)
        self.assertNotIn(PROBE_QUESTION, line)
        self.assertNotIn(PROBE_QUESTION, redact_access_target(raw_target))

    def test_filter_redacts_embedded_context_query(self):
        record = logging.LogRecord(
            name="app",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg=f"GET /retrieve/context?text={PROBE_QUESTION}&profile_stages=false",
            args=(),
            exc_info=None,
        )
        AccessLogPrivacyFilter().filter(record)
        message = record.getMessage()
        self.assertNotIn(PROBE_QUESTION, message)
        self.assertIn("text=[redacted]", message)

    def test_filter_redacts_literal_apostrophe_in_query_text(self):
        raw = "/retrieve/context?text=Alice's+neighbor+998877"
        redacted = redact_access_target(raw)
        self.assertIn("text=[redacted]", redacted)
        self.assertNotIn("Alice's", redacted)
        self.assertNotIn("998877", redacted)

        record = logging.LogRecord(
            name="uvicorn.access",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg='%s - "%s %s HTTP/%s" %d',
            args=("127.0.0.1:9", "GET", raw, "1.1", 200),
            exc_info=None,
        )
        AccessLogPrivacyFilter().filter(record)
        message = record.getMessage()
        self.assertIn("text=[redacted]", message)
        self.assertNotIn("Alice's", message)
        self.assertNotIn("998877", message)

    def test_filter_redacts_hash_in_query_text(self):
        raw = "/retrieve/context?text=Alice#998877"
        redacted = redact_access_target(raw)
        self.assertIn("text=[redacted]", redacted)
        self.assertNotIn("Alice", redacted)
        self.assertNotIn("998877", redacted)

    def test_filter_redacts_every_embedded_context_url(self):
        from src.services.api.logging_privacy import redact_embedded_message

        message = (
            "first /retrieve/context?text=FIRST "
            "second /retrieve/context?text=SECOND"
        )
        redacted = redact_embedded_message(message)
        self.assertEqual(redacted.count("text=[redacted]"), 2)
        self.assertNotIn("FIRST", redacted)
        self.assertNotIn("SECOND", redacted)

        record = logging.LogRecord(
            name="app",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg=message,
            args=(),
            exc_info=None,
        )
        AccessLogPrivacyFilter().filter(record)
        rendered = record.getMessage()
        self.assertEqual(rendered.count("text=[redacted]"), 2)
        self.assertNotIn("FIRST", rendered)
        self.assertNotIn("SECOND", rendered)

    def test_filter_redacts_percent_encoded_context_path(self):
        raw = "/retrieve/%63ontext?text=" + PROBE_QUESTION
        redacted = redact_access_target(raw)
        self.assertIn("text=[redacted]", redacted)
        self.assertNotIn(PROBE_QUESTION, redacted)

    def test_app_logger_record_is_redacted(self):
        install_access_log_privacy()
        logger = logging.getLogger("brainapi.test.privacy")
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
        previous_level = logger.level
        previous_propagate = logger.propagate
        logger.setLevel(logging.INFO)
        logger.propagate = False
        try:
            logger.info("GET /retrieve/context?text=%s", PROBE_QUESTION)
            line = stream.getvalue()
        finally:
            logger.removeHandler(handler)
            logger.setLevel(previous_level)
            logger.propagate = previous_propagate
        self.assertIn("text=[redacted]", line)
        self.assertNotIn(PROBE_QUESTION, line)


if __name__ == "__main__":
    unittest.main()
