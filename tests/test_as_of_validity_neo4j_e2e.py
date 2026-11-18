"""v1/v2/v3 as_of + since/until against the production Neo4j graph.

Skipped locally when Neo4j is unreachable. Required in Quality CI (the
workflow starts pipeline-neo4j and sets NEO4J_E2E_REQUIRED). Community
Neo4j only has the `neo4j` database. Chunks use Postgres when
PROGRESSIVE_POSTGRES_TEST_URI is set. No LLM is invoked.
"""

from __future__ import annotations

import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pytest

_TESTS = Path(__file__).resolve().parent
if str(_TESTS) not in sys.path:
    sys.path.insert(0, str(_TESTS))

from src.constants.data import TextChunk
from src.constants.kg import Node, Predicate
from src.core.saving.source_passages import screen_passage_hits
from src.core.saving.validity import parse_instant
from src.core.saving.validity_backfill import backfill_brain_validity
from src.core.saving.validity_changes import collect_validity_changes
from test_as_of_validity_e2e import (
    REG_V1,
    REG_V2,
    REG_V3,
    SOURCE_KEY,
    _configure_postgres,
    _drop_brain,
    _ingest_revision,
    _postgres_ready,
    _restore_postgres,
)

BRAIN_ID = "neo4j"


def _neo4j_required() -> bool:
    flag = os.getenv("NEO4J_E2E_REQUIRED", "")
    return flag.lower() in {"1", "true", "yes"} or os.getenv("CI") == "true"


def _apply_neo4j_env() -> None:
    from src.config import config

    config.neo4j.host = os.getenv("NEO4J_HOST", config.neo4j.host)
    config.neo4j.port = os.getenv("NEO4J_PORT", config.neo4j.port)
    config.neo4j.username = os.getenv("NEO4J_USERNAME", config.neo4j.username)
    config.neo4j.password = os.getenv("NEO4J_PASSWORD", config.neo4j.password)


def _neo4j_connectable() -> bool:
    _apply_neo4j_env()
    try:
        from neo4j import GraphDatabase
        from src.config import config

        driver = GraphDatabase.driver(
            f"bolt://{config.neo4j.host}:{config.neo4j.port}",
            auth=(config.neo4j.username, config.neo4j.password),
        )
        driver.verify_connectivity()
        driver.close()
        return True
    except Exception:
        return False


def _require_neo4j() -> None:
    if _neo4j_connectable():
        return
    if _neo4j_required():
        for _ in range(15):
            time.sleep(2)
            if _neo4j_connectable():
                return
        pytest.fail("Neo4j e2e is required but Neo4j is not reachable")
    pytest.skip("Neo4j is not available")


def _require_postgres() -> str:
    if _postgres_ready():
        return os.getenv("PROGRESSIVE_POSTGRES_TEST_URI")
    if _neo4j_required():
        pytest.fail("Neo4j as-of e2e requires PROGRESSIVE_POSTGRES_TEST_URI")
    pytest.skip("PostgreSQL data backend not available")


@pytest.fixture
def neo4j_as_of_stack():
    _require_neo4j()
    uri = _require_postgres()
    from src.adapters.data import DataAdapter
    from src.adapters.graph import GraphAdapter
    from src.lib.neo4j.client import Neo4jClient
    from src.lib.postgresql.data import PostgreSQLDataClient

    original = _configure_postgres(uri)
    client = Neo4jClient()
    graph = GraphAdapter()
    graph.add_client(client)
    data = DataAdapter()
    data.add_client(PostgreSQLDataClient())
    tracked: list[str] = []
    try:
        yield data, graph, client, tracked, uri
    finally:
        if tracked:
            try:
                client._run_query(
                    "MATCH (n) WHERE n['uuid'] IN $uuids DETACH DELETE n",
                    BRAIN_ID,
                    {"uuids": tracked},
                )
            except Exception:
                pass
        try:
            client.driver.close()
        except Exception:
            pass
        _drop_brain(uri, BRAIN_ID)
        _restore_postgres(original)


def test_neo4j_as_of_v1_v2_v3_regulation_and_changes_window(neo4j_as_of_stack):
    data, graph, client, tracked, _uri = neo4j_as_of_stack
    t1 = "2024-01-01T00:00:00Z"
    t2 = "2025-01-01T00:00:00Z"
    t3 = "2026-01-01T00:00:00Z"
    prefix = "asofn" + uuid4().hex[:8]
    subject_uuid = f"{prefix}-org"
    clause_uuid = f"{prefix}-clause"
    rel_v1 = f"{prefix}-rel-v1"
    rel_v2 = f"{prefix}-rel-v2"
    rel_v3 = f"{prefix}-rel-v3"
    tracked.extend([subject_uuid, clause_uuid])

    v1 = _ingest_revision(data, BRAIN_ID, REG_V1, t1, f"{prefix}-reg-v1")
    v2 = _ingest_revision(data, BRAIN_ID, REG_V2, t2, f"{prefix}-reg-v2")
    v3 = _ingest_revision(data, BRAIN_ID, REG_V3, t3, f"{prefix}-reg-v3")
    assert v1.id.endswith("reg-v1")

    subject = graph.add_nodes(
        [Node(uuid=subject_uuid, name="Issuer", labels=["ORG"])], BRAIN_ID
    )[0]
    clause = graph.add_nodes(
        [Node(uuid=clause_uuid, name="Article12", labels=["CLAUSE"])], BRAIN_ID
    )[0]
    graph.add_relationship(
        subject,
        Predicate(
            uuid=rel_v1,
            name="REQUIRES",
            description="five years",
            properties={"valid_from": t1, "valid_to": t2, "superseded_by": rel_v2},
        ),
        clause,
        BRAIN_ID,
    )
    graph.add_relationship(
        subject,
        Predicate(
            uuid=rel_v2,
            name="REQUIRES",
            description="seven years",
            properties={"valid_from": t2, "valid_to": t3, "superseded_by": rel_v3},
        ),
        clause,
        BRAIN_ID,
    )
    graph.add_relationship(
        subject,
        Predicate(
            uuid=rel_v3,
            name="REQUIRES",
            description="ten years",
            properties={"valid_from": t3},
        ),
        clause,
        BRAIN_ID,
    )

    stored, _ = data.get_text_chunks(
        brain_id=BRAIN_ID,
        limit=20,
        skip=0,
        metadata_eq={"source_key": SOURCE_KEY},
    )
    by_id = {str(chunk.id): chunk for chunk in stored}
    hits = [
        (chunk.id, 0.1, chunk.text)
        for chunk in (
            by_id[f"{prefix}-reg-v1"],
            by_id[f"{prefix}-reg-v2"],
            by_id[f"{prefix}-reg-v3"],
        )
    ]
    at_v1 = screen_passage_hits(
        data,
        BRAIN_ID,
        hits,
        include_history=False,
        as_of=parse_instant("2024-06-01T00:00:00Z"),
    )
    at_v2 = screen_passage_hits(
        data,
        BRAIN_ID,
        hits,
        include_history=False,
        as_of=parse_instant("2025-06-01T00:00:00Z"),
    )
    at_v3 = screen_passage_hits(
        data,
        BRAIN_ID,
        hits,
        include_history=False,
        as_of=parse_instant("2026-06-01T00:00:00Z"),
    )
    assert [item[2] for item in at_v1] == [REG_V1]
    assert [item[2] for item in at_v2] == [REG_V2]
    assert [item[2] for item in at_v3] == [REG_V3]
    live = screen_passage_hits(data, BRAIN_ID, hits, include_history=False)
    assert [item[2] for item in live] == [REG_V3]

    window = collect_validity_changes(
        data,
        graph,
        BRAIN_ID,
        since=parse_instant("2024-12-01T00:00:00Z"),
        until=parse_instant("2025-02-01T00:00:00Z"),
        limit=50,
        skip=0,
    )
    kinds = {(item["kind"], item["target"], item["id"]) for item in window["changes"]}
    assert ("superseded", "chunk", f"{prefix}-reg-v1") in kinds
    assert ("added", "chunk", f"{prefix}-reg-v2") in kinds
    assert ("superseded", "relationship", rel_v1) in kinds
    assert ("added", "relationship", rel_v2) in kinds

    listed, _ = graph.list_validity_relationships(
        BRAIN_ID,
        skip=0,
        limit=100,
        since="2024-12-01T00:00:00Z",
        until="2025-02-01T00:00:00Z",
    )
    listed_ids = {triple[1].uuid for triple in listed}
    assert rel_v1 in listed_ids
    assert rel_v2 in listed_ids


def test_neo4j_validity_backfill_mixed_dates_is_safe_twice(neo4j_as_of_stack):
    data, graph, _client, tracked, _uri = neo4j_as_of_stack
    prefix = "asofb" + uuid4().hex[:8]
    subject_uuid = f"{prefix}-org"
    clause_uuid = f"{prefix}-clause"
    rel_old = f"{prefix}-rel-old"
    rel_live = f"{prefix}-rel-live"
    tracked.extend([subject_uuid, clause_uuid])

    data.save_text_chunk(
        TextChunk(
            id=f"{prefix}-v1",
            text=REG_V1,
            metadata={"source_key": SOURCE_KEY, "superseded_at": "15/01/2025"},
            inserted_at=datetime(2024, 1, 1, 8, 0, 0),
        ),
        BRAIN_ID,
    )
    data.save_text_chunk(
        TextChunk(
            id=f"{prefix}-v2",
            text=REG_V2,
            metadata={"source_key": SOURCE_KEY, "valid_from": "2025-01-15T00:00:00Z"},
            inserted_at=datetime(2025, 1, 15, tzinfo=timezone.utc),
        ),
        BRAIN_ID,
    )
    subject = graph.add_nodes(
        [Node(uuid=subject_uuid, name="Issuer", labels=["ORG"])], BRAIN_ID
    )[0]
    clause = graph.add_nodes(
        [Node(uuid=clause_uuid, name="Article12", labels=["CLAUSE"])], BRAIN_ID
    )[0]
    graph.add_relationship(
        subject,
        Predicate(
            uuid=rel_old,
            name="REQUIRES",
            description="five years",
            properties={"valid_at": "01/01/2024", "invalid_at": "15/01/2025"},
        ),
        clause,
        BRAIN_ID,
    )
    graph.add_relationship(
        subject,
        Predicate(
            uuid=rel_live,
            name="REQUIRES",
            description="seven years",
            properties={"valid_at": "2025-01-15T00:00:00"},
        ),
        clause,
        BRAIN_ID,
    )
    first = backfill_brain_validity(data, graph, BRAIN_ID)
    assert first["chunks"] >= 1
    assert first["relationships"] >= 1
    stored, _ = data.get_text_chunks_by_ids(
        [f"{prefix}-v1", f"{prefix}-v2"], False, BRAIN_ID
    )
    by_id = {chunk.id: chunk for chunk in stored}
    assert by_id[f"{prefix}-v1"].text == REG_V1
    assert not by_id[f"{prefix}-v1"].metadata.get("valid_from")
    assert by_id[f"{prefix}-v1"].metadata["valid_to"] == "2025-01-15T00:00:00Z"
    assert by_id[f"{prefix}-v1"].metadata["superseded_at"] == "15/01/2025"
    assert not by_id[f"{prefix}-v2"].metadata.get("valid_to")

    triples, _ = graph.list_validity_relationships(BRAIN_ID, skip=0, limit=200)
    by_rel = {triple[1].uuid: triple[1].properties or {} for triple in triples}
    assert by_rel[rel_old].get("valid_from") == "2024-01-01T00:00:00Z"
    assert by_rel[rel_old].get("valid_to") == "2025-01-15T00:00:00Z"
    assert by_rel[rel_old].get("invalid_at") == "15/01/2025"
    assert by_rel[rel_live].get("valid_from") == "2025-01-15T00:00:00Z"
    assert not by_rel[rel_live].get("valid_to")

    dry = backfill_brain_validity(data, graph, BRAIN_ID, dry_run=True)
    assert dry["chunks"] == 0
    assert dry["relationships"] == 0
    second = backfill_brain_validity(data, graph, BRAIN_ID)
    assert second["chunks"] == 0
    assert second["relationships"] == 0
