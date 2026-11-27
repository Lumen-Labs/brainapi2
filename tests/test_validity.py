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

from src.constants.data import Observation, TextChunk
from src.constants.embeddings import Vector
from src.constants.kg import Node, Predicate
from src.core.saving.validity import (
    InvalidAsOf,
    VALIDITY_UNKNOWN,
    VALIDITY_VALID,
    change_kind,
    chunk_is_valid_at,
    close_validity,
    dedupe_unknown_items,
    format_instant,
    is_valid_at,
    optional_instant,
    parse_instant,
    predicate_is_valid_at,
    stamp_valid_from,
    unknown_fact,
    unknown_relationship_id,
    validity_status,
)
from src.core.saving.validity_backfill import (
    backfill_brain_validity,
    backfill_chunk_validity,
)
from src.core.saving.validity_cardinality import (
    should_close_outgoing_edge,
    single_valued_relationship_types,
)
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

    def test_as_of_omits_undated_copies_and_live_view_keeps_them(self):
        as_of = _utc("2024-06-01T00:00:00Z")
        self.assertTrue(is_valid_at({}, None))
        self.assertFalse(is_valid_at({}, as_of))
        self.assertFalse(
            is_valid_at(
                {"recorded_at": "2024-01-01T00:00:00Z"},
                as_of,
            )
        )
        self.assertFalse(
            is_valid_at({}, as_of, inserted_at=datetime(2024, 1, 1, tzinfo=timezone.utc))
        )
        pred = Predicate(uuid="r-undated", name="WORKS_WITH", description="", properties={})
        self.assertTrue(predicate_is_valid_at(pred, None))
        self.assertFalse(predicate_is_valid_at(pred, as_of))
        self.assertEqual(validity_status({}, None), "valid")
        self.assertEqual(validity_status({}, as_of), VALIDITY_UNKNOWN)
        self.assertEqual(
            validity_status({"recorded_at": "2024-01-01T00:00:00Z"}, as_of),
            VALIDITY_UNKNOWN,
        )
        self.assertEqual(
            unknown_fact(target="chunk", item_id="plain")["validity"],
            VALIDITY_UNKNOWN,
        )
        dated = Predicate(
            uuid="r-dated",
            name="REQUIRES",
            description="",
            properties={"valid_from": "2024-01-01T00:00:00Z"},
        )
        undated = Predicate(
            uuid="r-undated",
            name="MENTIONS",
            description="",
            properties={},
        )
        self.assertEqual(
            unknown_relationship_id(
                dated, undated, VALIDITY_VALID, VALIDITY_UNKNOWN
            ),
            "r-undated",
        )
        self.assertEqual(
            unknown_relationship_id(
                undated, undated, VALIDITY_UNKNOWN, VALIDITY_UNKNOWN
            ),
            "r-undated|r-undated",
        )
        duplicated = [
            unknown_fact(target="chunk", item_id="plain", payload="a"),
            unknown_fact(target="chunk", item_id="plain", payload="a"),
            unknown_fact(target="chunk", item_id="plain", payload="b"),
        ]
        self.assertEqual(len(dedupe_unknown_items(duplicated)), 2)

    def test_as_of_before_ingest_uses_valid_from_not_recorded_at(self):
        fields = {
            "valid_from": "2020-01-01T00:00:00Z",
            "recorded_at": "2026-11-18T12:00:00Z",
        }
        self.assertTrue(is_valid_at(fields, _utc("2021-06-01T00:00:00Z")))
        undated = {"recorded_at": "2026-11-18T12:00:00Z"}
        self.assertFalse(is_valid_at(undated, _utc("2021-06-01T00:00:00Z")))
        self.assertFalse(is_valid_at(undated, _utc("2026-12-01T00:00:00Z")))
        self.assertTrue(is_valid_at(undated, None))

    def test_legacy_stamps_are_read(self):
        chunk = TextChunk(
            id="c1",
            text="old",
            metadata={
                "valid_from": "2026-01-01T00:00:00Z",
                "superseded_at": "15/11/2026",
            },
            inserted_at=datetime(2026, 1, 1),
        )
        self.assertFalse(chunk_is_valid_at(chunk, None))
        self.assertTrue(
            chunk_is_valid_at(chunk, _utc("2026-11-14T00:00:00Z"))
        )
        self.assertFalse(
            chunk_is_valid_at(
                TextChunk(
                    id="c-undated",
                    text="old",
                    metadata={"superseded_at": "15/11/2026"},
                    inserted_at=datetime(2026, 1, 1),
                ),
                _utc("2026-11-14T00:00:00Z"),
            )
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
                {"recorded_at": "2025-06-01T00:00:00Z"},
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
        slash_stamp = "15/01/2025"
        iso_since = "2025-01-01T00:00:00Z"
        iso_until = "2025-02-01T00:00:00Z"
        self.assertEqual(
            change_kind(
                {"invalid_at": slash_stamp},
                _utc(iso_since),
                _utc(iso_until),
            ),
            "invalidated",
        )
        self.assertFalse(
            slash_stamp >= iso_since and slash_stamp < iso_until
        )

    def test_stamp_and_close_are_idempotent(self):
        first = stamp_valid_from({}, "2024-01-01T00:00:00Z")
        second = stamp_valid_from(first, "2025-01-01T00:00:00Z")
        self.assertEqual(first["valid_from"], second["valid_from"])
        undated = stamp_valid_from(
            {}, None, recorded_at="2026-11-18T12:00:00Z"
        )
        self.assertNotIn("valid_from", undated)
        self.assertEqual(undated["recorded_at"], "2026-11-18T12:00:00Z")
        sourced = stamp_valid_from(
            {}, "2020-01-01T00:00:00Z", recorded_at="2026-11-18T12:00:00Z"
        )
        self.assertEqual(sourced["valid_from"], "2020-01-01T00:00:00Z")
        self.assertEqual(sourced["recorded_at"], "2026-11-18T12:00:00Z")
        closed = close_validity(
            first,
            closed_at="2025-01-01T00:00:00Z",
            superseded_by="n2",
            reason="single-valued predicate replaced",
        )
        again = close_validity(closed, closed_at="2026-01-01T00:00:00Z", superseded_by="n3")
        self.assertEqual(closed["valid_to"], again["valid_to"])
        self.assertEqual(closed["superseded_by"], "n2")
        self.assertEqual(closed["retired_reason"], "single-valued predicate replaced")
        self.assertEqual(again["retired_reason"], "single-valued predicate replaced")
        messy = close_validity({}, closed_at="15/01/2025")
        self.assertEqual(messy["invalid_at"], "2025-01-15T00:00:00Z")
        self.assertEqual(messy["valid_to"], "2025-01-15T00:00:00Z")


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
        self.assertNotIn("valid_from", chunk.metadata)
        self.assertEqual(chunk.metadata["valid_to"], "2025-01-01T00:00:00Z")
        self.assertEqual(backfill_chunk_validity(store, "brain"), 0)
        self.assertEqual(len(saved), 1)

    def test_mixed_legacy_copy_fixture_is_idempotent_and_leaves_live_copies(self):
        v1 = TextChunk(
            id="reg-v1",
            text="Article 12. five years.",
            metadata={
                "source_key": "reg-article-12",
                "superseded_at": "15/01/2025",
            },
            inserted_at=datetime(2024, 1, 1, 8, 0, 0),
        )
        v2 = TextChunk(
            id="reg-v2",
            text="Article 12. seven years.",
            metadata={
                "source_key": "reg-article-12",
                "superseded_at": "2026-01-01",
            },
            inserted_at=datetime(2025, 1, 15, tzinfo=timezone.utc),
        )
        v3 = TextChunk(
            id="reg-v3",
            text="Article 12. ten years.",
            metadata={
                "source_key": "reg-article-12",
                "valid_from": "2026-01-01T00:00:00Z",
            },
            inserted_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        )
        live = TextChunk(
            id="unrelated",
            text="keep me live",
            metadata={"note": "no source_key"},
            inserted_at=datetime(2024, 6, 1, tzinfo=timezone.utc),
        )
        chunks = [v1, v2, v3, live]
        saved = []

        class Store:
            def get_text_chunks(self, brain_id, limit=10, skip=0, order="asc"):
                if skip:
                    return [], len(chunks)
                return chunks, len(chunks)

            def save_text_chunk(self, item, brain_id):
                saved.append((item.id, dict(item.metadata), item.text))

        old_rel = Predicate(
            uuid="rel-old",
            name="REQUIRES",
            description="five years",
            properties={
                "valid_at": "01/01/2024",
                "invalid_at": "15/01/2025",
                "source_chunk_ids": ["reg-v1"],
            },
            deprecated=True,
        )
        live_rel = Predicate(
            uuid="rel-live",
            name="WORKS_WITH",
            description="colleague",
            properties={"valid_at": "2024-06-01T12:00:00", "source_chunk_ids": ["unrelated"]},
        )
        already = Predicate(
            uuid="rel-done",
            name="REQUIRES",
            description="ten years",
            properties={"valid_from": "2026-01-01T00:00:00Z", "source_chunk_ids": ["reg-v3"]},
        )
        unknown_start = Predicate(
            uuid="rel-unknown",
            name="REQUIRES",
            description="closed without start",
            properties={"invalid_at": "15/01/2025"},
        )
        triples = [
            (Node(uuid="org", name="Issuer", labels=["ORG"]), old_rel, Node(uuid="c", name="Clause", labels=["CLAUSE"])),
            (Node(uuid="alice", name="Alice", labels=["PERSON"]), live_rel, Node(uuid="bob", name="Bob", labels=["PERSON"])),
            (Node(uuid="org", name="Issuer", labels=["ORG"]), already, Node(uuid="c", name="Clause", labels=["CLAUSE"])),
            (Node(uuid="org", name="Issuer", labels=["ORG"]), unknown_start, Node(uuid="c", name="Clause", labels=["CLAUSE"])),
        ]

        class Graph:
            def list_validity_relationships(self, brain_id, skip=0, limit=100, since=None, until=None):
                if skip:
                    return [], len(triples)
                return triples, len(triples)

            def update_properties(self, uuid, kind, brain_id, new_properties, properties_to_remove=None):
                for _, pred, _ in triples:
                    if pred.uuid == uuid:
                        pred.properties = {**(pred.properties or {}), **(new_properties or {})}

        store = Store()
        graph = Graph()
        first = backfill_brain_validity(store, graph, "brain")
        self.assertEqual(first["chunks"], 2)
        self.assertEqual(first["relationships"], 3)
        self.assertNotIn("valid_from", v1.metadata)
        self.assertEqual(v1.metadata["valid_to"], "2025-01-15T00:00:00Z")
        self.assertEqual(v1.metadata["superseded_at"], "15/01/2025")
        self.assertEqual(v2.metadata["valid_to"], "2026-01-01T00:00:00Z")
        self.assertEqual(v3.metadata["valid_from"], "2026-01-01T00:00:00Z")
        self.assertNotIn("valid_to", v3.metadata)
        self.assertEqual(live.metadata, {"note": "no source_key"})
        self.assertEqual(old_rel.properties["valid_from"], "2024-01-01T00:00:00Z")
        self.assertEqual(old_rel.properties["valid_to"], "2025-01-15T00:00:00Z")
        self.assertEqual(old_rel.properties["invalid_at"], "15/01/2025")
        self.assertEqual(live_rel.properties["valid_from"], "2024-06-01T12:00:00Z")
        self.assertNotIn("valid_to", live_rel.properties)
        self.assertEqual(already.properties, {
            "valid_from": "2026-01-01T00:00:00Z",
            "source_chunk_ids": ["reg-v3"],
        })
        self.assertEqual(unknown_start.properties["valid_to"], "2025-01-15T00:00:00Z")
        self.assertNotIn("valid_from", unknown_start.properties)
        self.assertEqual({row[0] for row in saved}, {"reg-v1", "reg-v2"})
        self.assertEqual(saved[0][2], "Article 12. five years.")
        dry = backfill_brain_validity(store, graph, "brain", dry_run=True)
        self.assertEqual(dry, {"chunks": 0, "relationships": 0})
        second = backfill_brain_validity(store, graph, "brain")
        self.assertEqual(second, {"chunks": 0, "relationships": 0})
        self.assertEqual(len(saved), 2)


class ValidityCardinalityTests(unittest.TestCase):
    def test_multi_valued_types_are_not_closed_unless_source_revision(self):
        self.assertFalse(
            should_close_outgoing_edge(
                relationship_name="WORKS_WITH",
                existing_properties={"source_chunk_ids": ["c1"]},
            )
        )
        self.assertFalse(
            should_close_outgoing_edge(
                relationship_name="APPLIES_TO",
                existing_properties={},
            )
        )
        self.assertTrue(
            should_close_outgoing_edge(
                relationship_name="LIVES_IN",
                existing_properties={},
            )
        )
        self.assertFalse(
            should_close_outgoing_edge(
                relationship_name="HEADQUARTERED_IN",
                existing_properties={"source_key": "profile.txt"},
                incoming_source_key="press.txt",
                require_same_source=True,
            )
        )
        self.assertFalse(
            should_close_outgoing_edge(
                relationship_name="HEADQUARTERED_IN",
                existing_properties={},
                incoming_source_key="press.txt",
                require_same_source=True,
            )
        )
        self.assertTrue(
            should_close_outgoing_edge(
                relationship_name="HEADQUARTERED_IN",
                existing_properties={"source_key": "profile.txt"},
                incoming_source_key="profile.txt",
                require_same_source=True,
            )
        )
        self.assertFalse(
            should_close_outgoing_edge(
                relationship_name="REQUIRES",
                existing_properties={},
            )
        )
        self.assertFalse(
            should_close_outgoing_edge(
                relationship_name="LOCATED_IN",
                existing_properties={},
            )
        )
        self.assertTrue(
            should_close_outgoing_edge(
                relationship_name="WORKS_WITH",
                existing_properties={"source_chunk_ids": ["old"]},
                superseded_chunk_ids=["old"],
            )
        )

    def test_allowlist_env_extends_single_valued_types(self):
        self.assertNotIn("APPLIES_TO", single_valued_relationship_types())
        self.assertNotIn("REQUIRES", single_valued_relationship_types())
        with patch.dict(os.environ, {"VALIDITY_SINGLE_VALUED_REL_TYPES": "APPLIES_TO,REQUIRES"}):
            self.assertTrue(
                should_close_outgoing_edge(
                    relationship_name="APPLIES_TO",
                    existing_properties={},
                )
            )
            self.assertTrue(
                should_close_outgoing_edge(
                    relationship_name="REQUIRES",
                    existing_properties={},
                )
            )


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

    def test_changelog_rows_format_model_timestamps(self):
        class Adapter:
            def get_text_chunks(self, brain_id, limit=10, skip=0, order="desc"):
                return [], 0

            def get_changelogs_list(self, *args, **kwargs):
                return [
                    SimpleNamespace(
                        id="chg-1",
                        type="relationship_created",
                        timestamp=datetime(2025, 1, 15, 12, 0, tzinfo=timezone.utc),
                    )
                ]

        payload = collect_validity_changes(
            Adapter(),
            None,
            "brain",
            since=_utc("2025-01-01T00:00:00Z"),
            until=_utc("2025-02-01T00:00:00Z"),
            limit=10,
            skip=0,
        )
        self.assertEqual(payload["changes"][0]["timestamp"], "2025-01-15T12:00:00Z")

    def test_merged_changelog_page_is_not_capped_at_200(self):
        rows = [
            SimpleNamespace(
                id=f"chg-{index}",
                type="relationship_created",
                timestamp=datetime(2025, 1, 1, tzinfo=timezone.utc),
            )
            for index in range(250)
        ]

        class Adapter:
            def get_text_chunks(self, brain_id, limit=10, skip=0, order="desc"):
                return [], 0

            def get_changelogs_list(
                self, brain_id, limit=10, skip=0, types=None, query_text=None, since=None, until=None
            ):
                return rows[skip : skip + limit]

        payload = collect_validity_changes(
            Adapter(),
            None,
            "brain",
            since=_utc("2024-12-01T00:00:00Z"),
            until=_utc("2025-02-01T00:00:00Z"),
            limit=10,
            skip=200,
        )
        self.assertEqual(payload["total"], 250)
        self.assertEqual(payload["count"], 10)
        self.assertEqual(payload["changes"][0]["id"], "chg-200")

    def test_relationship_window_is_passed_through_and_pages(self):
        calls = []
        rels = [
            (
                Node(uuid="a", name="Org", labels=["ORG"]),
                Predicate(
                    uuid=f"rel-{index}",
                    name="REQUIRES",
                    description="",
                    properties={"valid_from": "2025-01-01T00:00:00Z"},
                ),
                Node(uuid="b", name="Retention", labels=["CLAUSE"]),
            )
            for index in range(3)
        ]

        class Adapter:
            def get_text_chunks(self, brain_id, limit=10, skip=0, order="desc"):
                return [], 0

            def get_changelogs_list(self, *args, **kwargs):
                return []

        class Graph:
            def list_validity_relationships(
                self, brain_id, skip=0, limit=100, since=None, until=None
            ):
                calls.append(
                    {"skip": skip, "limit": limit, "since": since, "until": until}
                )
                return rels[skip : skip + limit], len(rels)

        payload = collect_validity_changes(
            Adapter(),
            Graph(),
            "brain",
            since=_utc("2024-12-01T00:00:00Z"),
            until=_utc("2025-02-01T00:00:00Z"),
            limit=1,
            skip=1,
        )
        self.assertEqual(payload["total"], 3)
        self.assertEqual(payload["count"], 1)
        self.assertEqual(payload["changes"][0]["id"], "rel-1")
        self.assertTrue(calls)
        self.assertIsNotNone(calls[0]["since"])
        self.assertIsNotNone(calls[0]["until"])
        self.assertEqual(calls[0]["skip"], 0)

    def test_changelog_typeerror_fallback_stays_inside_the_window(self):
        class Adapter:
            def get_text_chunks(self, brain_id, limit=10, skip=0, order="desc"):
                return [], 0

            def get_changelogs_list(self, *args):
                if len(args) > 5:
                    raise TypeError("old adapter")
                return [
                    SimpleNamespace(
                        id="in-window",
                        type="relationship_created",
                        timestamp=datetime(2025, 1, 15, tzinfo=timezone.utc),
                    ),
                    SimpleNamespace(
                        id="out-of-window",
                        type="relationship_created",
                        timestamp=datetime(2023, 1, 1, tzinfo=timezone.utc),
                    ),
                ]

        payload = collect_validity_changes(
            Adapter(),
            None,
            "brain",
            since=_utc("2025-01-01T00:00:00Z"),
            until=_utc("2025-02-01T00:00:00Z"),
            limit=10,
            skip=0,
        )
        self.assertEqual([item["id"] for item in payload["changes"]], ["in-window"])

    def test_backfill_pages_past_ten_thousand(self):
        seen = []
        dummy = TextChunk(id="x", text="t", metadata={"valid_from": "2024-01-01T00:00:00Z"})

        class Store:
            def get_text_chunks(self, brain_id, limit=10, skip=0, order="asc"):
                seen.append(skip)
                total = 10050
                if skip >= total:
                    return [], total
                return [dummy] * min(limit, total - skip), total

            def save_text_chunk(self, item, brain_id):
                raise AssertionError("already stamped")

        self.assertEqual(backfill_chunk_validity(Store(), "brain"), 0)
        self.assertIn(10000, seen)

    def test_chunk_fill_keeps_concurrent_end_stamp(self):
        meta = {"superseded_at": "2025-01-01T00:00:00Z"}

        class Store:
            def get_text_chunks(self, brain_id, limit=10, skip=0, order="asc"):
                chunk = TextChunk(
                    id="c1",
                    text="t",
                    metadata=dict(meta),
                    inserted_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
                )
                return [chunk], 1

            def fill_text_chunk_validity(self, chunk_id, brain_id, patch):
                if not meta.get("valid_to"):
                    meta["valid_to"] = "concurrent"
                for key, value in patch.items():
                    if not meta.get(key):
                        meta[key] = value

            def save_text_chunk(self, item, brain_id):
                raise AssertionError("use fill_text_chunk_validity")

        self.assertEqual(backfill_chunk_validity(Store(), "brain"), 1)
        self.assertNotIn("valid_from", meta)
        self.assertEqual(meta["valid_to"], "concurrent")
        self.assertEqual(backfill_chunk_validity(Store(), "brain"), 0)


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
        from src.adapters.interfaces.data import SearchResult
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
            Predicate(
                uuid="r-live-2",
                name="FOR",
                description="",
                properties={"valid_from": "2025-01-01T00:00:00Z"},
            ),
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
            Predicate(
                uuid="r-old-2",
                name="FOR",
                description="",
                properties={
                    "valid_from": "2024-01-01T00:00:00Z",
                    "valid_to": "2025-01-01T00:00:00Z",
                },
            ),
            Node(uuid="tip", name="FiveYears", labels=["CLAUSE"]),
        )
        undated = (
            Node(uuid="tail", name="Org", labels=["ORG"]),
            Predicate(
                uuid="r-undated",
                name="MENTIONS",
                description="",
                properties={},
            ),
            Node(uuid="mid", name="Clause12", labels=["EVENT"]),
            Predicate(
                uuid="r-undated-2",
                name="FOR",
                description="",
                properties={},
            ),
            Node(uuid="tip", name="UndatedNote", labels=["NOTE"]),
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
        mock_graph.get_event_centric_neighbors.return_value = [live, old, undated]
        mock_graph.get_hub_bridges.return_value = []
        mock_graph.list_topic_memberships.return_value = []
        mock_data = MagicMock()
        mock_data.get_last_text_chunks.return_value = []
        mock_data.get_last_structured_data.return_value = []
        mock_data.get_text_chunks_by_ids.return_value = ([], [])
        mock_data.search.return_value = SearchResult(text_chunks=[], observations=[])
        mock_data.search_bm25.return_value = []
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
            as_of_shown = await retrieve_mod.get_context(
                GetContextRequestBody(
                    text="retention clause",
                    brain_id="brain",
                    as_of="2024-06-01T00:00:00Z",
                    include_validity_unknown=True,
                )
            )
            bad = await self._bad_as_of(retrieve_mod)
        self.assertIn("SevenYears", live_response.text_context)
        self.assertNotIn("FiveYears", live_response.text_context)
        self.assertIn("UndatedNote", live_response.text_context)
        self.assertIsNone(live_response.validity_unknown_count)
        self.assertIn("FiveYears", as_of_response.text_context)
        self.assertNotIn("SevenYears", as_of_response.text_context)
        self.assertNotIn("UndatedNote", as_of_response.text_context)
        self.assertNotIn("[superseded]", as_of_response.text_context)
        self.assertEqual(as_of_response.validity_unknown_count, 1)
        self.assertIsNone(as_of_response.validity_unknown)
        self.assertEqual(as_of_shown.validity_unknown_count, 1)
        self.assertEqual(as_of_shown.validity_unknown[0]["validity"], "unknown")
        self.assertEqual(as_of_shown.validity_unknown[0]["id"], "r-undated|r-undated-2")
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
        undated = TextChunk(id="plain", text="no dates", metadata={})
        mock_embeddings = MagicMock()
        mock_embeddings.embed_text.return_value = Vector(
            id="q", embeddings=[0.1], metadata={}
        )
        mock_vs = MagicMock()
        mock_vs.search_data.return_value = []
        mock_vs.search_triplets.return_value = []
        mock_data = MagicMock()
        mock_data.search.return_value = SearchResult(
            text_chunks=[v1, v2, undated],
            observations=[
                Observation(id="o1", text="five", resource_id="v1"),
                Observation(id="o2", text="seven", resource_id="v2"),
            ],
        )
        mock_data.get_text_chunks_by_ids.return_value = ([], [])
        mock_graph = MagicMock()
        mock_graph.nodes_and_edges_citing.return_value = ([], [])
        mock_graph.get_nodes_by_uuid.return_value = [
            {
                "node": Node(uuid="org", name="Org", labels=["ORG"]),
                "relationships": {
                    "valid_from": "2025-01-01T00:00:00Z",
                    "name": "REQUIRES",
                },
                "related_nodes": Node(uuid="clause", name="SevenYears", labels=["CLAUSE"]),
            },
            {
                "node": Node(uuid="org", name="Org", labels=["ORG"]),
                "relationships": {"name": "MENTIONS"},
                "related_nodes": Node(uuid="note", name="UndatedNote", labels=["NOTE"]),
            },
        ]
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
            as_of_shown = await retrieve_mod.retrieve_data(
                "years",
                10,
                "",
                "brain",
                as_of="2024-06-01T00:00:00Z",
                include_validity_unknown=True,
            )
            later = await retrieve_mod.retrieve_data(
                "years", 10, "", "brain", as_of="2025-06-01T00:00:00Z"
            )
        self.assertEqual({chunk.id for chunk in current.data}, {"v1", "v2", "plain"})
        self.assertEqual({chunk.id for chunk in as_of.data}, {"v1"})
        self.assertEqual({chunk.id for chunk in later.data}, {"v2"})
        self.assertEqual({obs.resource_id for obs in current.observations}, {"v1", "v2"})
        self.assertEqual({obs.resource_id for obs in as_of.observations}, {"v1"})
        self.assertEqual({obs.resource_id for obs in later.observations}, {"v2"})
        self.assertEqual(len(current.relationships), 2)
        self.assertEqual(as_of.relationships, [])
        self.assertEqual(len(later.relationships), 1)
        self.assertIsNone(current.validity_unknown_count)
        self.assertEqual(as_of.validity_unknown_count, 2)
        self.assertIsNone(as_of.validity_unknown)
        self.assertEqual(as_of_shown.validity_unknown_count, 2)
        unknown_ids = {item["id"] for item in as_of_shown.validity_unknown}
        self.assertEqual(unknown_ids, {"plain", "MENTIONS"})
        self.assertTrue(
            all(item["validity"] == "unknown" for item in as_of_shown.validity_unknown)
        )


class McpAsOfUnknownTests(unittest.TestCase):
    def test_search_semantically_as_of_reports_unknown(self):
        from src.constants.embeddings import Vector
        from src.services.mcp import main as mcp_mod

        dated = (
            Node(uuid="tail", name="Org", labels=["ORG"]),
            Predicate(
                uuid="r-dated",
                name="REQUIRES",
                description="",
                properties={"valid_from": "2024-01-01T00:00:00Z"},
            ),
            Node(uuid="mid", name="Clause12", labels=["EVENT"]),
            Predicate(
                uuid="r-dated-2",
                name="FOR",
                description="",
                properties={"valid_from": "2024-01-01T00:00:00Z"},
            ),
            Node(uuid="tip", name="FiveYears", labels=["CLAUSE"]),
        )
        undated = (
            Node(uuid="tail", name="Org", labels=["ORG"]),
            Predicate(
                uuid="r-undated",
                name="MENTIONS",
                description="",
                properties={},
            ),
            Node(uuid="mid", name="Clause12", labels=["EVENT"]),
            Predicate(
                uuid="r-undated-2",
                name="FOR",
                description="",
                properties={},
            ),
            Node(uuid="tip", name="UndatedNote", labels=["NOTE"]),
        )
        mock_embeddings = MagicMock()
        mock_embeddings.embed_text.return_value = Vector(
            id="q", embeddings=[0.1], metadata={}
        )
        mock_vs = MagicMock()
        mock_vs.search_nodes.return_value = [
            Vector(id="n1", embeddings=[0.1], metadata={"uuid": "tail"})
        ]
        mock_graph = MagicMock()
        mock_graph.get_event_centric_neighbors.return_value = [dated, undated]
        with (
            patch.object(mcp_mod, "guard_brainpat", return_value=True),
            patch.object(mcp_mod, "embeddings_adapter", mock_embeddings),
            patch.object(mcp_mod, "vector_search", mock_vs),
            patch.object(mcp_mod, "graph_adapter", mock_graph),
        ):
            live = mcp_mod._search_semantically_sync("q", "brain", False)
            hidden = mcp_mod._search_semantically_sync(
                "q", "brain", False, as_of="2024-06-01T00:00:00Z"
            )
            shown = mcp_mod._search_semantically_sync(
                "q",
                "brain",
                False,
                as_of="2024-06-01T00:00:00Z",
                include_validity_unknown=True,
            )
        self.assertEqual(len(live), 2)
        self.assertEqual(len(hidden["results"]), 1)
        self.assertEqual(hidden["results"][0][1].uuid, "r-dated")
        self.assertEqual(hidden["validity_unknown_count"], 1)
        self.assertNotIn("validity_unknown", hidden)
        self.assertEqual(shown["validity_unknown_count"], 1)
        self.assertEqual(shown["validity_unknown"][0]["validity"], "unknown")
        self.assertEqual(shown["validity_unknown"][0]["id"], "r-undated|r-undated-2")
        self.assertIsInstance(shown["validity_unknown"][0]["payload"], list)

    def test_search_semantically_as_of_uses_unknown_rel_uuid(self):
        from src.constants.embeddings import Vector
        from src.services.mcp import main as mcp_mod

        mixed = (
            Node(uuid="tail", name="Org", labels=["ORG"]),
            Predicate(
                uuid="r-dated",
                name="REQUIRES",
                description="",
                properties={"valid_from": "2024-01-01T00:00:00Z"},
            ),
            Node(uuid="mid", name="Clause12", labels=["EVENT"]),
            Predicate(
                uuid="r-undated",
                name="MENTIONS",
                description="",
                properties={},
            ),
            Node(uuid="tip", name="UndatedNote", labels=["NOTE"]),
        )
        mock_embeddings = MagicMock()
        mock_embeddings.embed_text.return_value = Vector(
            id="q", embeddings=[0.1], metadata={}
        )
        mock_vs = MagicMock()
        mock_vs.search_nodes.return_value = [
            Vector(id="n1", embeddings=[0.1], metadata={"uuid": "tail"})
        ]
        mock_graph = MagicMock()
        mock_graph.get_event_centric_neighbors.return_value = [mixed]
        with (
            patch.object(mcp_mod, "guard_brainpat", return_value=True),
            patch.object(mcp_mod, "embeddings_adapter", mock_embeddings),
            patch.object(mcp_mod, "vector_search", mock_vs),
            patch.object(mcp_mod, "graph_adapter", mock_graph),
        ):
            shown = mcp_mod._search_semantically_sync(
                "q",
                "brain",
                False,
                as_of="2024-06-01T00:00:00Z",
                include_validity_unknown=True,
            )
        self.assertEqual(shown["validity_unknown_count"], 1)
        self.assertEqual(shown["validity_unknown"][0]["id"], "r-undated")


class RetrieveUnknownJsonTests(unittest.TestCase):
    def test_retrieve_response_json_encodes_unknown_payloads(self):
        from src.services.api.constants.requests import RetrieveRequestResponse

        class _Blob:
            def __str__(self):
                return "blob"

        response = RetrieveRequestResponse(
            data=[],
            observations=[],
            relationships=[],
            validity_unknown_count=1,
            validity_unknown=[
                {
                    "validity": "unknown",
                    "target": "relationship",
                    "id": "r1",
                    "payload": {"relationships": _Blob()},
                }
            ],
        )
        dumped = response.model_dump(mode="json")
        self.assertEqual(
            dumped["validity_unknown"][0]["payload"]["relationships"],
            "blob",
        )


class BridgeUnknownAsOfTests(unittest.TestCase):
    def test_bridge_expand_reports_undated_facts(self):
        from src.core.saving.hub_bridges import HubBridge
        from src.services.api.controllers import retrieve as retrieve_mod

        seed = {
            "identified_entity": "q",
            "triple": (
                Node(uuid="n", name="n", labels=["PERSON"], properties={}),
                Predicate(
                    uuid="r1",
                    name="MADE",
                    description="",
                    properties={"valid_from": "2024-01-01T00:00:00Z"},
                ),
                Node(
                    uuid="event-seed",
                    name="event-seed",
                    labels=["EVENT"],
                    properties={},
                ),
                Predicate(
                    uuid="r2",
                    name="TARGETED",
                    description="",
                    properties={"valid_from": "2024-01-01T00:00:00Z"},
                ),
                Node(uuid="b", name="b", labels=["PERSON"], properties={}),
            ),
            "score": 0.1,
            "key": ("r1", "r2"),
            "text": "fact-event-seed",
            "chunk_ids": [],
        }
        bridged = (
            Node(uuid="actor2", name="actor2", labels=["PERSON"], properties={}),
            Predicate(uuid="r3", name="MADE", description="", properties={}),
            Node(
                uuid="event-bridge",
                name="event-bridge",
                labels=["EVENT"],
                properties={},
            ),
            Predicate(uuid="r4", name="TARGETED", description="", properties={}),
            Node(uuid="obj2", name="obj2", labels=["PERSON"], properties={}),
        )
        mock_adapter = MagicMock()
        mock_adapter.get_hub_bridges.return_value = [
            HubBridge(
                event_a="event-bridge",
                event_b="event-seed",
                shared_entity="actor",
                shared_entity_name="actor",
                weight=1.0,
            )
        ]
        mock_adapter.get_event_hub_facts.return_value = [bridged]
        unknown = []
        with patch.object(retrieve_mod, "graph_adapter", mock_adapter):
            expanded, _paths = retrieve_mod._expand_cross_event_bridges(
                [seed],
                "brain",
                max_per_hub=3,
                as_of=_utc("2024-06-01T00:00:00Z"),
                unknown_out=unknown,
            )
        self.assertEqual(len(expanded), 1)
        self.assertEqual(expanded[0]["key"], ("r1", "r2"))
        self.assertEqual(len(unknown), 1)
        self.assertEqual(unknown[0]["validity"], "unknown")
        self.assertEqual(unknown[0]["id"], "r3|r4")


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
