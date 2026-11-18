"""Point-in-time retrieve against the real PostgreSQL stack used in Quality CI.

No LLM is invoked: ingest uses skip_enrichment-style chunk writes plus structured
graph adapter writes. Neo4j is optional when a server is reachable.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from urllib.parse import urlparse
from uuid import uuid4

import pytest

from src.constants.data import TextChunk
from src.constants.kg import Node, Predicate
from src.core.saving.source_passages import prepare_source_revision, screen_passage_hits
from src.core.saving.validity import format_instant, parse_instant
from src.core.saving.validity_changes import collect_validity_changes


REG_V1 = "Article 12. Retention of records shall be five years from the date of issue."
REG_V2 = "Article 12. Retention of records shall be seven years from the date of issue."
REG_V3 = "Article 12. Retention of records shall be ten years from the date of issue."
SOURCE_KEY = "reg-article-12"


def _postgres_ready() -> bool:
    uri = os.getenv("PROGRESSIVE_POSTGRES_TEST_URI")
    if not uri:
        return False
    try:
        import psycopg2

        conn = psycopg2.connect(uri)
        conn.close()
        return True
    except Exception:
        return False


def _configure_postgres(uri: str):
    from src.config import config

    parsed = urlparse(uri)
    original = (
        config.postgresql.host,
        config.postgresql.port,
        config.postgresql.username,
        config.postgresql.password,
        config.postgresql.database,
        config.postgresql.system_database,
        config.postgresql.maintenance_database,
    )
    config.postgresql.host = parsed.hostname
    config.postgresql.port = parsed.port or 5432
    config.postgresql.username = parsed.username
    config.postgresql.password = parsed.password
    dbname = (parsed.path or "/postgres").lstrip("/") or "postgres"
    config.postgresql.database = dbname
    config.postgresql.system_database = dbname
    config.postgresql.maintenance_database = dbname
    return original


def _restore_postgres(original):
    from src.config import config

    (
        config.postgresql.host,
        config.postgresql.port,
        config.postgresql.username,
        config.postgresql.password,
        config.postgresql.database,
        config.postgresql.system_database,
        config.postgresql.maintenance_database,
    ) = original


def _drop_brain(uri: str, brain_id: str) -> None:
    try:
        from src.lib.postgresql._provisioning import drop_brain_database

        drop_brain_database(brain_id)
    except Exception:
        pass
    try:
        import psycopg2
        from psycopg2 import sql
        from src.lib.postgresql._naming import brain_db_name

        admin = psycopg2.connect(uri)
        admin.autocommit = True
        with admin.cursor() as cur:
            cur.execute(
                sql.SQL("DROP DATABASE IF EXISTS {}").format(
                    sql.Identifier(brain_db_name(brain_id))
                )
            )
        admin.close()
    except Exception:
        pass


def _ingest_revision(adapter, brain_id: str, text: str, stamp: str, chunk_id: str) -> TextChunk:
    meta = {"source_key": SOURCE_KEY}
    superseded_ids: list[str] = []
    prepare_source_revision(
        adapter,
        brain_id=brain_id,
        source_text=text,
        meta_keys=meta,
        source_timestamp=stamp,
        out_superseded=superseded_ids,
    )
    chunk = TextChunk(
        id=chunk_id,
        text=text,
        metadata=dict(meta),
        inserted_at=parse_instant(stamp).replace(tzinfo=None)
        if parse_instant(stamp)
        else datetime.now(timezone.utc).replace(tzinfo=None),
    )
    saved = adapter.save_text_chunk(chunk, brain_id)
    if superseded_ids:
        adapter.mark_text_chunks_superseded(
            superseded_ids,
            stamp,
            brain_id,
            superseded_by=chunk_id,
            valid_to=format_instant(parse_instant(stamp)),
        )
    return saved


@pytest.mark.skipif(not _postgres_ready(), reason="PostgreSQL data backend not available")
def test_as_of_v1_v2_v3_regulation_and_changes_window():
    from src.adapters.data import DataAdapter
    from src.adapters.graph import GraphAdapter
    from src.lib.postgresql.data import PostgreSQLDataClient
    from src.lib.postgresql.networkx_client import NetworkXGraphClient
    from src.services.api.controllers import retrieve as retrieve_mod
    from src.services.api.controllers.changelogs import get_changelogs_list
    from unittest.mock import MagicMock, patch
    from src.constants.embeddings import Vector
    import asyncio

    uri = os.getenv("PROGRESSIVE_POSTGRES_TEST_URI")
    original = _configure_postgres(uri)
    brain_id = "asof" + uuid4().hex[:10]
    data = DataAdapter()
    data.add_client(PostgreSQLDataClient())
    graph = GraphAdapter()
    graph.add_client(NetworkXGraphClient())
    t1 = "2024-01-01T00:00:00Z"
    t2 = "2025-01-01T00:00:00Z"
    t3 = "2026-01-01T00:00:00Z"
    try:
        v1 = _ingest_revision(data, brain_id, REG_V1, t1, "reg-v1")
        v2 = _ingest_revision(data, brain_id, REG_V2, t2, "reg-v2")
        v3 = _ingest_revision(data, brain_id, REG_V3, t3, "reg-v3")
        assert v1.id == "reg-v1"
        subject = graph.add_nodes(
            [Node(uuid="org-1", name="Issuer", labels=["ORG"])], brain_id
        )[0]
        clause = graph.add_nodes(
            [Node(uuid="clause-12", name="Article12", labels=["CLAUSE"])],
            brain_id,
        )[0]
        r1 = Predicate(
            uuid="rel-v1",
            name="REQUIRES",
            description="five years",
            properties={"valid_from": t1, "valid_to": t2, "superseded_by": "rel-v2"},
        )
        r2 = Predicate(
            uuid="rel-v2",
            name="REQUIRES",
            description="seven years",
            properties={"valid_from": t2, "valid_to": t3, "superseded_by": "rel-v3"},
        )
        r3 = Predicate(
            uuid="rel-v3",
            name="REQUIRES",
            description="ten years",
            properties={"valid_from": t3},
        )
        graph.add_relationship(subject, r1, clause, brain_id)
        graph.add_relationship(subject, r2, clause, brain_id)
        graph.add_relationship(subject, r3, clause, brain_id)

        stored, _ = data.get_text_chunks(
            brain_id=brain_id,
            limit=20,
            skip=0,
            metadata_eq={"source_key": SOURCE_KEY},
        )
        by_id = {str(chunk.id): chunk for chunk in stored}
        assert "valid_from" in (by_id["reg-v1"].metadata or {})
        assert (by_id["reg-v1"].metadata or {}).get("valid_to")
        assert (by_id["reg-v1"].metadata or {}).get("superseded_by") == "reg-v2"
        assert not (by_id["reg-v3"].metadata or {}).get("valid_to")

        hits = [
            (chunk.id, 0.1, chunk.text)
            for chunk in (by_id["reg-v1"], by_id["reg-v2"], by_id["reg-v3"])
        ]
        at_v1 = screen_passage_hits(
            data,
            brain_id,
            hits,
            include_history=False,
            as_of=parse_instant("2024-06-01T00:00:00Z"),
        )
        at_v2 = screen_passage_hits(
            data,
            brain_id,
            hits,
            include_history=False,
            as_of=parse_instant("2025-06-01T00:00:00Z"),
        )
        at_v3 = screen_passage_hits(
            data,
            brain_id,
            hits,
            include_history=False,
            as_of=parse_instant("2026-06-01T00:00:00Z"),
        )
        assert [item[2] for item in at_v1] == [REG_V1]
        assert [item[2] for item in at_v2] == [REG_V2]
        assert [item[2] for item in at_v3] == [REG_V3]

        live = screen_passage_hits(data, brain_id, hits, include_history=False)
        assert [item[2] for item in live] == [REG_V3]

        window = collect_validity_changes(
            data,
            graph,
            brain_id,
            since=parse_instant("2024-12-01T00:00:00Z"),
            until=parse_instant("2025-02-01T00:00:00Z"),
            limit=50,
            skip=0,
        )
        kinds = {(item["kind"], item["target"], item["id"]) for item in window["changes"]}
        assert ("superseded", "chunk", "reg-v1") in kinds
        assert ("added", "chunk", "reg-v2") in kinds

        mock_embeddings = MagicMock()
        mock_embeddings.embed_text.return_value = Vector(
            id="q", embeddings=[0.1], metadata={}
        )
        mock_vs = MagicMock()
        mock_vs.search_data.return_value = []
        mock_vs.search_triplets.return_value = []
        mock_graph = MagicMock()
        mock_graph.get_nodes_by_uuid.return_value = []

        async def _retrieve(as_of=None):
            with (
                patch.object(retrieve_mod, "embeddings_adapter", mock_embeddings),
                patch.object(retrieve_mod, "vector_search", mock_vs),
                patch.object(retrieve_mod, "graph_adapter", mock_graph),
                patch.object(retrieve_mod, "data_adapter", data),
            ):
                return await retrieve_mod.retrieve_data(
                    "Article 12", 10, "", brain_id, as_of=as_of
                )

        as_of_v1 = asyncio.run(_retrieve("2024-06-01T00:00:00Z"))
        as_of_v3 = asyncio.run(_retrieve("2026-06-01T00:00:00Z"))
        live_retrieve = asyncio.run(_retrieve(None))
        assert {chunk.text for chunk in as_of_v1.data} == {REG_V1}
        assert {chunk.text for chunk in as_of_v3.data} == {REG_V3}
        assert REG_V3 in {chunk.text for chunk in live_retrieve.data}

        async def _changes():
            with (
                patch(
                    "src.services.api.controllers.changelogs.data_adapter",
                    data,
                ),
                patch(
                    "src.services.api.controllers.changelogs.graph_adapter",
                    graph,
                ),
            ):
                return await get_changelogs_list(
                    limit=50,
                    skip=0,
                    brain_id=brain_id,
                    since="2024-12-01T00:00:00Z",
                    until="2025-02-01T00:00:00Z",
                )

        response = asyncio.run(_changes())
        body = response.body.decode("utf-8")
        assert "reg-v1" in body
        assert "reg-v2" in body
        assert "superseded" in body
    finally:
        _drop_brain(uri, brain_id)
        _restore_postgres(original)


@pytest.mark.skipif(not _postgres_ready(), reason="PostgreSQL data backend not available")
def test_staged_validity_mark_rolls_back_until_replay():
    from src.adapters.data import DataAdapter
    from src.core.saving.ingest_publication import replay_data_operation
    from src.core.saving.ingest_write_buffer import capture_writes
    from src.lib.postgresql.data import PostgreSQLDataClient

    uri = os.getenv("PROGRESSIVE_POSTGRES_TEST_URI")
    original = _configure_postgres(uri)
    brain_id = "asofstg" + uuid4().hex[:8]
    data = DataAdapter()
    data.add_client(PostgreSQLDataClient())
    try:
        chunk = TextChunk(
            id="keep-me",
            text=REG_V1,
            metadata={
                "source_key": SOURCE_KEY,
                "valid_from": "2024-01-01T00:00:00Z",
            },
        )
        data.save_text_chunk(chunk, brain_id)
        with capture_writes(brain_id) as operations:
            data.mark_text_chunks_superseded(
                ["keep-me"],
                "2025-01-01T00:00:00Z",
                brain_id,
                superseded_by="next",
                valid_to="2025-01-01T00:00:00Z",
            )
        loaded, _ = data.get_text_chunks_by_ids(["keep-me"], False, brain_id)
        assert not (loaded[0].metadata or {}).get("superseded_at")
        assert operations[0]["method"] == "mark_text_chunks_superseded"
        replay_data_operation(data.data, brain_id, operations[0])
        loaded, _ = data.get_text_chunks_by_ids(["keep-me"], False, brain_id)
        assert (loaded[0].metadata or {}).get("superseded_at") == "2025-01-01T00:00:00Z"
        assert (loaded[0].metadata or {}).get("valid_to") == "2025-01-01T00:00:00Z"
        assert (loaded[0].metadata or {}).get("superseded_by") == "next"
    finally:
        _drop_brain(uri, brain_id)
        _restore_postgres(original)


@pytest.mark.skipif(not _postgres_ready(), reason="PostgreSQL data backend not available")
def test_validity_backfill_mixed_legacy_dates_on_live_postgres_brain():
    from src.adapters.data import DataAdapter
    from src.adapters.graph import GraphAdapter
    from src.core.saving.validity_backfill import backfill_brain_validity
    from src.lib.postgresql.data import PostgreSQLDataClient
    from src.lib.postgresql.networkx_client import NetworkXGraphClient

    uri = os.getenv("PROGRESSIVE_POSTGRES_TEST_URI")
    original = _configure_postgres(uri)
    brain_id = "asofbf" + uuid4().hex[:8]
    data = DataAdapter()
    data.add_client(PostgreSQLDataClient())
    graph = GraphAdapter()
    graph.add_client(NetworkXGraphClient())
    try:
        data.save_text_chunk(
            TextChunk(
                id="reg-v1",
                text=REG_V1,
                metadata={"source_key": SOURCE_KEY, "superseded_at": "15/01/2025"},
                inserted_at=datetime(2024, 1, 1, 8, 0, 0),
            ),
            brain_id,
        )
        data.save_text_chunk(
            TextChunk(
                id="reg-v2",
                text=REG_V2,
                metadata={"source_key": SOURCE_KEY, "valid_from": "2025-01-15T00:00:00Z"},
                inserted_at=datetime(2025, 1, 15, tzinfo=timezone.utc),
            ),
            brain_id,
        )
        subject = graph.add_nodes(
            [Node(uuid="org-bf", name="Issuer", labels=["ORG"])], brain_id
        )[0]
        clause = graph.add_nodes(
            [Node(uuid="clause-bf", name="Article12", labels=["CLAUSE"])], brain_id
        )[0]
        graph.add_relationship(
            subject,
            Predicate(
                uuid="rel-old",
                name="REQUIRES",
                description="five years",
                properties={"valid_at": "01/01/2024", "invalid_at": "15/01/2025"},
            ),
            clause,
            brain_id,
        )
        graph.add_relationship(
            subject,
            Predicate(
                uuid="rel-live",
                name="REQUIRES",
                description="seven years",
                properties={"valid_at": "2025-01-15T00:00:00"},
            ),
            clause,
            brain_id,
        )
        first = backfill_brain_validity(data, graph, brain_id)
        assert first["chunks"] >= 1
        assert first["relationships"] >= 1
        stored, _ = data.get_text_chunks_by_ids(["reg-v1", "reg-v2"], False, brain_id)
        by_id = {chunk.id: chunk for chunk in stored}
        assert by_id["reg-v1"].text == REG_V1
        assert by_id["reg-v1"].metadata["valid_from"] == "2024-01-01T08:00:00Z"
        assert by_id["reg-v1"].metadata["valid_to"] == "2025-01-15T00:00:00Z"
        assert by_id["reg-v1"].metadata["superseded_at"] == "15/01/2025"
        assert by_id["reg-v2"].metadata["valid_from"] == "2025-01-15T00:00:00Z"
        assert not by_id["reg-v2"].metadata.get("valid_to")
        dry = backfill_brain_validity(data, graph, brain_id, dry_run=True)
        assert dry == {"chunks": 0, "relationships": 0}
        second = backfill_brain_validity(data, graph, brain_id)
        assert second == {"chunks": 0, "relationships": 0}
    finally:
        _drop_brain(uri, brain_id)
        _restore_postgres(original)
