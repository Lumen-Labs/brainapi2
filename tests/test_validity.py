import os
import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from fastapi import HTTPException

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

from src.constants.data import TextChunk
from src.constants.embeddings import Vector
from src.constants.kg import Node, Predicate
from src.core.saving.validity import (
    InvalidAsOf,
    change_kind,
    chunk_is_valid_at,
    close_validity,
    format_instant,
    is_valid_at,
    optional_instant,
    parse_instant,
    predicate_is_valid_at,
    stamp_valid_from,
)
from src.core.saving.validity_backfill import backfill_chunk_validity
from src.core.saving.validity_changes import collect_validity_changes
from src.services.api.constants.requests import GetContextRequestBody
from src.utils.nlp.ner import ExtractElementsResponse


def _utc(text: str) -> datetime:
    return parse_instant(text)


class ValidityIntervalTests(unittest.TestCase):
    def test_half_open_interval(self):
        fields = {
            "valid_from": "2024-01-01T00:00:00Z",
            "valid_to": "2025-01-01T00:00:00Z",
        }
        self.assertTrue(is_valid_at(fields, _utc("2024-01-01T00:00:00Z")))
        self.assertTrue(is_valid_at(fields, _utc("2024-12-31T23:59:59Z")))
        self.assertFalse(is_valid_at(fields, _utc("2025-01-01T00:00:00Z")))
        self.assertFalse(is_valid_at(fields, _utc("2023-12-31T23:59:59Z")))

    def test_live_view_hides_closed_and_deprecated(self):
        self.assertTrue(is_valid_at({"valid_from": "2024-01-01T00:00:00Z"}, None))
        self.assertFalse(
            is_valid_at(
                {
                    "valid_from": "2024-01-01T00:00:00Z",
                    "valid_to": "2025-01-01T00:00:00Z",
                },
                None,
            )
        )
        self.assertFalse(is_valid_at({}, None, deprecated=True))

    def test_legacy_stamps_are_read(self):
        chunk = TextChunk(
            id="c1",
            text="old",
            metadata={"superseded_at": "15/11/2026"},
            inserted_at=datetime(2026, 1, 1),
        )
        self.assertFalse(chunk_is_valid_at(chunk, None))
        self.assertTrue(
            chunk_is_valid_at(chunk, _utc("2026-11-14T00:00:00Z"))
        )
        pred = Predicate(
            uuid="r1",
            name="OWNS",
            description="",
            properties={"invalid_at": "2025-06-01T00:00:00Z", "valid_at": "2024-01-01T00:00:00Z"},
            deprecated=True,
        )
        self.assertFalse(predicate_is_valid_at(pred, None))
        self.assertTrue(predicate_is_valid_at(pred, _utc("2025-01-01T00:00:00Z")))

    def test_optional_instant_rejects_garbage(self):
        self.assertIsNone(optional_instant(None))
        self.assertIsNone(optional_instant("  "))
        with self.assertRaises(InvalidAsOf):
            optional_instant("not-a-date")
        self.assertEqual(format_instant(_utc("2024-02-03T04:05:06Z")), "2024-02-03T04:05:06Z")

    def test_change_kind_added_superseded_invalidated(self):
        since = _utc("2025-01-01T00:00:00Z")
        until = _utc("2026-01-01T00:00:00Z")
        self.assertEqual(
            change_kind(
                {"valid_from": "2025-06-01T00:00:00Z"},
                since,
                until,
            ),
            "added",
        )
        self.assertEqual(
            change_kind(
                {
                    "valid_from": "2024-01-01T00:00:00Z",
                    "valid_to": "2025-06-01T00:00:00Z",
                    "superseded_by": "next",
                },
                since,
                until,
            ),
            "superseded",
        )
        self.assertEqual(
            change_kind(
                {
                    "valid_from": "2024-01-01T00:00:00Z",
                    "invalid_at": "2025-06-01T00:00:00Z",
                },
                since,
                until,
            ),
            "invalidated",
        )

    def test_stamp_and_close_are_idempotent(self):
        first = stamp_valid_from({}, "2024-01-01T00:00:00Z")
        second = stamp_valid_from(first, "2025-01-01T00:00:00Z")
        self.assertEqual(first["valid_from"], second["valid_from"])
        closed = close_validity(first, closed_at="2025-01-01T00:00:00Z", superseded_by="n2")
        again = close_validity(closed, closed_at="2026-01-01T00:00:00Z", superseded_by="n3")
        self.assertEqual(closed["valid_to"], again["valid_to"])
        self.assertEqual(closed["superseded_by"], "n2")


class ValidityBackfillTests(unittest.TestCase):
    def test_backfill_is_idempotent(self):
        chunk = TextChunk(
            id="c1",
            text="hello",
            metadata={"superseded_at": "2025-01-01T00:00:00Z"},
            inserted_at=datetime(2024, 6, 1, tzinfo=timezone.utc),
        )
        saved = []

        class Store:
            def get_text_chunks(self, brain_id, limit=10, skip=0, order="asc"):
                return [chunk], 1

            def save_text_chunk(self, item, brain_id):
                saved.append(dict(item.metadata))

        store = Store()
        self.assertEqual(backfill_chunk_validity(store, "brain"), 1)
        self.assertEqual(chunk.metadata["valid_from"], "2024-06-01T00:00:00Z")
        self.assertEqual(chunk.metadata["valid_to"], "2025-01-01T00:00:00Z")
        self.assertEqual(backfill_chunk_validity(store, "brain"), 0)
        self.assertEqual(len(saved), 1)


class ValidityChangesTests(unittest.TestCase):
    def test_collects_chunk_and_relationship_changes(self):
        v1 = TextChunk(
            id="v1",
            text="five years",
            metadata={
                "source_key": "reg-12",
                "valid_from": "2024-01-01T00:00:00Z",
                "valid_to": "2025-01-01T00:00:00Z",
                "superseded_by": "v2",
                "superseded_at": "2025-01-01T00:00:00Z",
            },
        )
        v2 = TextChunk(
            id="v2",
            text="seven years",
            metadata={
                "source_key": "reg-12",
                "valid_from": "2025-01-01T00:00:00Z",
            },
        )

        class Adapter:
            def get_text_chunks(self, brain_id, limit=10, skip=0, order="desc"):
                if skip:
                    return [], 2
                return [v2, v1], 2

            def get_changelogs_list(self, *args, **kwargs):
                return []

        class Graph:
            def list_validity_relationships(self, brain_id, skip=0, limit=100, since=None, until=None):
                if skip:
                    return [], 1
                pred = Predicate(
                    uuid="rel-1",
                    name="REQUIRES",
                    description="",
                    properties={
                        "valid_from": "2025-01-01T00:00:00Z",
                        "source_chunk_ids": ["v2"],
                    },
                )
                return [
                    (
                        Node(uuid="a", name="Org", labels=["ORG"]),
                        pred,
                        Node(uuid="b", name="Retention", labels=["CLAUSE"]),
                    )
                ], 1

        payload = collect_validity_changes(
            Adapter(),
            Graph(),
            "brain",
            since=_utc("2024-12-01T00:00:00Z"),
            until=_utc("2025-02-01T00:00:00Z"),
            limit=10,
            skip=0,
        )
        kinds = {(item["kind"], item["target"], item["id"]) for item in payload["changes"]}
        self.assertIn(("superseded", "chunk", "v1"), kinds)
        self.assertIn(("added", "chunk", "v2"), kinds)
        self.assertIn(("added", "relationship", "rel-1"), kinds)
        self.assertEqual(payload["count"], 3)


class StagedValidityWriteTests(unittest.TestCase):
    def test_mark_is_captured_and_replayed(self):
        from src.adapters.data import DataAdapter
        from src.core.saving.ingest_publication import replay_data_operation
        from src.core.saving.ingest_write_buffer import capture_writes

        client = MagicMock()
        adapter = DataAdapter()
        adapter.add_client(client)
        with capture_writes("brain") as operations:
            adapter.mark_text_chunks_superseded(
                ["c1"],
                "2025-01-01T00:00:00Z",
                "brain",
                superseded_by="c2",
                valid_to="2025-01-01T00:00:00Z",
            )
        client.mark_text_chunks_superseded.assert_not_called()
        self.assertEqual(operations[0]["method"], "mark_text_chunks_superseded")
        self.assertEqual(operations[0]["args"]["superseded_by"], "c2")
        replay_data_operation(client, "brain", operations[0])
        client.mark_text_chunks_superseded.assert_called_once()


class AsOfRetrieveTests(unittest.IsolatedAsyncioTestCase):
    async def test_get_context_as_of_keeps_then_current_fact_without_prefix(self):
        from src.services.api.controllers import retrieve as retrieve_mod

        live = (
            Node(uuid="tail", name="Org", labels=["ORG"]),
            Predicate(
                uuid="r-live",
                name="REQUIRES",
                description="",
                properties={"valid_from": "2025-01-01T00:00:00Z"},
            ),
            Node(uuid="mid", name="Clause12", labels=["EVENT"]),
            Predicate(uuid="r-live-2", name="FOR", description="", properties={}),
            Node(uuid="tip", name="SevenYears", labels=["CLAUSE"]),
        )
        old = (
            Node(uuid="tail", name="Org", labels=["ORG"]),
            Predicate(
                uuid="r-old",
                name="REQUIRES",
                description="",
                properties={
                    "valid_from": "2024-01-01T00:00:00Z",
                    "valid_to": "2025-01-01T00:00:00Z",
                    "superseded_by": "r-live",
                    "deprecated": True,
                },
                deprecated=True,
            ),
            Node(uuid="mid", name="Clause12", labels=["EVENT"]),
            Predicate(uuid="r-old-2", name="FOR", description="", properties={}),
            Node(uuid="tip", name="FiveYears", labels=["CLAUSE"]),
        )
        mock_embeddings = MagicMock()
        mock_embeddings.embed_text.return_value = Vector(
            id="q", embeddings=[0.1, 0.2, 0.3], metadata={}
        )
        mock_vs = MagicMock()
        mock_vs.search_nodes.return_value = [
            Vector(
                id="n1",
                embeddings=[0.1, 0.2, 0.3],
                metadata={"uuid": "tail", "name": "Org"},
                distance=0.1,
            )
        ]
        mock_vs.search_relationships.return_value = []
        mock_vs.search_data.return_value = []
        mock_graph = MagicMock()
        mock_graph.get_event_centric_neighbors.return_value = [live, old]
        mock_graph.get_hub_bridges.return_value = []
        mock_graph.list_topic_memberships.return_value = []
        mock_data = MagicMock()
        mock_data.get_last_text_chunks.return_value = []
        mock_data.get_last_structured_data.return_value = []
        mock_data.get_text_chunks_by_ids.return_value = ([], [])
        mock_data.search.side_effect = Exception("unused")
        mock_data.search_bm25.side_effect = Exception("unused")
        elements = ExtractElementsResponse(tokens=[], noun_chunks=[])
        with (
            patch.object(retrieve_mod, "embeddings_adapter", mock_embeddings),
            patch.object(retrieve_mod, "vector_search", mock_vs),
            patch.object(retrieve_mod, "graph_adapter", mock_graph),
            patch.object(retrieve_mod, "data_adapter", mock_data),
            patch.object(
                retrieve_mod._entity_extractor,
                "extract_elements",
                return_value=elements,
            ),
        ):
            live_response = await retrieve_mod.get_context(
                GetContextRequestBody(text="retention clause", brain_id="brain")
            )
            as_of_response = await retrieve_mod.get_context(
                GetContextRequestBody(
                    text="retention clause",
                    brain_id="brain",
                    as_of="2024-06-01T00:00:00Z",
                )
            )
            bad = await self._bad_as_of(retrieve_mod)
        self.assertIn("SevenYears", live_response.text_context)
        self.assertNotIn("FiveYears", live_response.text_context)
        self.assertIn("FiveYears", as_of_response.text_context)
        self.assertNotIn("SevenYears", as_of_response.text_context)
        self.assertNotIn("[superseded]", as_of_response.text_context)
        self.assertEqual(bad.status_code, 400)

    async def _bad_as_of(self, retrieve_mod):
        try:
            await retrieve_mod.get_context(
                GetContextRequestBody(
                    text="retention clause",
                    brain_id="brain",
                    as_of="not-a-date",
                )
            )
        except HTTPException as exc:
            return SimpleNamespace(status_code=exc.status_code)
        return SimpleNamespace(status_code=200)

    async def test_retrieve_data_filters_chunks_by_as_of(self):
        from src.services.api.controllers import retrieve as retrieve_mod
        from src.adapters.interfaces.data import SearchResult

        v1 = TextChunk(
            id="v1",
            text="five years",
            metadata={
                "valid_from": "2024-01-01T00:00:00Z",
                "valid_to": "2025-01-01T00:00:00Z",
            },
        )
        v2 = TextChunk(
            id="v2",
            text="seven years",
            metadata={"valid_from": "2025-01-01T00:00:00Z"},
        )
        mock_embeddings = MagicMock()
        mock_embeddings.embed_text.return_value = Vector(
            id="q", embeddings=[0.1], metadata={}
        )
        mock_vs = MagicMock()
        mock_vs.search_data.return_value = []
        mock_vs.search_triplets.return_value = []
        mock_data = MagicMock()
        mock_data.search.return_value = SearchResult(
            text_chunks=[v1, v2], observations=[]
        )
        mock_data.get_text_chunks_by_ids.return_value = ([], [])
        mock_graph = MagicMock()
        mock_graph.get_nodes_by_uuid.return_value = []
        with (
            patch.object(retrieve_mod, "embeddings_adapter", mock_embeddings),
            patch.object(retrieve_mod, "vector_search", mock_vs),
            patch.object(retrieve_mod, "data_adapter", mock_data),
            patch.object(retrieve_mod, "graph_adapter", mock_graph),
        ):
            current = await retrieve_mod.retrieve_data("years", 10, "", "brain")
            as_of = await retrieve_mod.retrieve_data(
                "years", 10, "", "brain", as_of="2024-06-01T00:00:00Z"
            )
            later = await retrieve_mod.retrieve_data(
                "years", 10, "", "brain", as_of="2025-06-01T00:00:00Z"
            )
        self.assertEqual({chunk.id for chunk in current.data}, {"v1", "v2"})
        self.assertEqual({chunk.id for chunk in as_of.data}, {"v1"})
        self.assertEqual({chunk.id for chunk in later.data}, {"v2"})


class ChangelogWindowControllerTests(unittest.IsolatedAsyncioTestCase):
    async def test_since_until_returns_unified_changes(self):
        from src.services.api.controllers import changelogs as changelogs_mod

        payload = {
            "changes": [
                {
                    "kind": "added",
                    "target": "chunk",
                    "id": "v2",
                    "timestamp": "2025-01-01T00:00:00Z",
                }
            ],
            "count": 1,
            "total": 1,
            "since": "2024-12-01T00:00:00Z",
            "until": "2025-02-01T00:00:00Z",
        }
        with (
            patch.object(
                changelogs_mod,
                "collect_validity_changes",
                return_value=payload,
            ),
            patch.object(changelogs_mod, "data_adapter") as data_adapter,
            patch.object(changelogs_mod, "graph_adapter"),
        ):
            data_adapter.get_changelogs_list.return_value = []
            response = await changelogs_mod.get_changelogs_list(
                limit=10,
                skip=0,
                types=None,
                query_text=None,
                brain_id="brain",
                since="2024-12-01T00:00:00Z",
                until="2025-02-01T00:00:00Z",
            )
        body = response.body
        self.assertIn(b"added", body)
        self.assertIn(b"v2", body)

    async def test_invalid_since_is_400(self):
        from src.services.api.controllers import changelogs as changelogs_mod

        with self.assertRaises(HTTPException) as raised:
            await changelogs_mod.get_changelogs_list(
                since="nope", until="2025-01-01T00:00:00Z"
            )
        self.assertEqual(raised.exception.status_code, 400)


if __name__ == "__main__":
    unittest.main()
