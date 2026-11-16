"""Cross-source conflict detection against the real PostgreSQL stack used in Quality CI.

No LLM is invoked. A seeded help-center corpus is written as chunks plus graph
triples (skip_enrichment-style). Detection, REST, retrieval annotation, and
the scan-job body all run against that brain.
"""

from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime, timezone
from unittest.mock import patch
from urllib.parse import urlparse
from uuid import uuid4

import pytest

from src.constants.data import TextChunk
from src.constants.kg import Node, Predicate
from src.core.saving.conflicts import (
    detect_conflicts,
    maybe_detect_after_ingest,
    open_conflicts_by_fact_ids,
)
from src.core.saving.validity import parse_instant


EXPECTED_PAIRS = {
    frozenset({"fact-refund-v1", "fact-refund-14"}),
    frozenset({"fact-refund-v2", "fact-refund-14"}),
    frozenset({"fact-fee-yes", "fact-fee-no"}),
    frozenset({"fact-policy-full", "fact-policy-credit"}),
}
FORBIDDEN_PAIRS = {
    frozenset({"fact-refund-v1", "fact-refund-v2"}),
    frozenset({"fact-feature-chat", "fact-feature-email"}),
    frozenset({"fact-ship-a", "fact-ship-b"}),
}


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


def _chunk(chunk_id: str, text: str, source_key: str, stamp: str) -> TextChunk:
    inserted = parse_instant(stamp)
    return TextChunk(
        id=chunk_id,
        text=text,
        metadata={
            "source_key": source_key,
            "title": source_key,
            "valid_from": stamp,
        },
        inserted_at=inserted.replace(tzinfo=None) if inserted else datetime.now(
            timezone.utc
        ).replace(tzinfo=None),
    )


def _rel(uuid: str, name: str, stamp: str, source_key: str, chunk_id: str, extra=None):
    props = {
        "valid_from": stamp,
        "source_key": source_key,
        "source_chunk_ids": [chunk_id],
        "source_timestamp": stamp,
    }
    if extra:
        props.update(extra)
    return Predicate(uuid=uuid, name=name, description="", properties=props)


def _seed(data, graph, brain_id: str):
    t1 = "2024-01-01T00:00:00Z"
    t2 = "2025-01-01T00:00:00Z"
    chunks = [
        _chunk(
            "chunk-refunds-v1",
            "Refunds are accepted within 30 days of purchase.",
            "help/refunds",
            t1,
        ),
        _chunk(
            "chunk-refunds-v2",
            "Refunds are accepted within 60 days of purchase.",
            "help/refunds",
            t2,
        ),
        _chunk(
            "chunk-faq",
            "You have 14 days to request a refund.",
            "help/faq",
            t1,
        ),
        _chunk(
            "chunk-features",
            "The product includes live chat and email support.",
            "help/features",
            t1,
        ),
        _chunk(
            "chunk-fees",
            "A restocking fee applies to opened items.",
            "help/fees",
            t1,
        ),
        _chunk(
            "chunk-returns",
            "A restocking fee does not apply to opened items.",
            "help/returns",
            t1,
        ),
        _chunk(
            "chunk-shipping",
            "Free shipping on orders over $50.",
            "help/shipping",
            t1,
        ),
        _chunk(
            "chunk-checkout",
            "Orders over $50 ship free.",
            "help/checkout",
            t1,
        ),
        _chunk(
            "chunk-policy-a",
            "Customers receive a full refund.",
            "help/policy-a",
            t1,
        ),
        _chunk(
            "chunk-policy-b",
            "Customers receive store credit only.",
            "help/policy-b",
            t1,
        ),
    ]
    for chunk in chunks:
        data.save_text_chunk(chunk, brain_id)

    policy = graph.add_nodes(
        [Node(uuid="policy-1", name="RefundPolicy", labels=["POLICY"])],
        brain_id,
    )[0]
    days30 = graph.add_nodes(
        [Node(uuid="win-30", name="30 days", labels=["DURATION"])], brain_id
    )[0]
    days14 = graph.add_nodes(
        [Node(uuid="win-14", name="14 days", labels=["DURATION"])], brain_id
    )[0]
    days60 = graph.add_nodes(
        [Node(uuid="win-60", name="60 days", labels=["DURATION"])], brain_id
    )[0]
    chat = graph.add_nodes(
        [Node(uuid="feat-chat", name="chat", labels=["FEATURE"])], brain_id
    )[0]
    email = graph.add_nodes(
        [Node(uuid="feat-email", name="email", labels=["FEATURE"])], brain_id
    )[0]
    fee_yes = graph.add_nodes(
        [
            Node(
                uuid="fee-yes-node",
                name="restocking fee",
                labels=["FEE"],
                polarity="positive",
                properties={"polarity": "positive"},
            )
        ],
        brain_id,
    )[0]
    fee_no = graph.add_nodes(
        [
            Node(
                uuid="fee-no-node",
                name="restocking fee",
                labels=["FEE"],
                polarity="negative",
                properties={"polarity": "negative"},
            )
        ],
        brain_id,
    )[0]
    ship = graph.add_nodes(
        [Node(uuid="ship-50", name="free over $50", labels=["OFFER"])], brain_id
    )[0]
    full = graph.add_nodes(
        [Node(uuid="full-refund", name="full refund", labels=["POLICY_VALUE"])],
        brain_id,
    )[0]
    credit = graph.add_nodes(
        [Node(uuid="store-credit", name="store credit only", labels=["POLICY_VALUE"])],
        brain_id,
    )[0]

    graph.add_relationship(
        policy,
        _rel(
            "fact-refund-v1",
            "HAS_WINDOW",
            t1,
            "help/refunds",
            "chunk-refunds-v1",
            extra={"valid_to": t2, "superseded_by": "fact-refund-v2"},
        ),
        days30,
        brain_id,
    )
    graph.add_relationship(
        policy,
        _rel("fact-refund-v2", "HAS_WINDOW", t2, "help/refunds", "chunk-refunds-v2"),
        days60,
        brain_id,
    )
    graph.add_relationship(
        policy,
        _rel("fact-refund-14", "HAS_WINDOW", t1, "help/faq", "chunk-faq"),
        days14,
        brain_id,
    )
    graph.add_relationship(
        policy,
        _rel("fact-feature-chat", "HAS_FEATURE", t1, "help/features", "chunk-features"),
        chat,
        brain_id,
    )
    graph.add_relationship(
        policy,
        _rel("fact-feature-email", "HAS_FEATURE", t1, "help/features", "chunk-features"),
        email,
        brain_id,
    )
    graph.add_relationship(
        policy,
        _rel(
            "fact-fee-yes",
            "APPLIES",
            t1,
            "help/fees",
            "chunk-fees",
            extra={"polarity": "positive"},
        ),
        fee_yes,
        brain_id,
    )
    graph.add_relationship(
        policy,
        _rel(
            "fact-fee-no",
            "APPLIES",
            t1,
            "help/returns",
            "chunk-returns",
            extra={"polarity": "negative"},
        ),
        fee_no,
        brain_id,
    )
    graph.add_relationship(
        policy,
        _rel("fact-ship-a", "STATES", t1, "help/shipping", "chunk-shipping"),
        ship,
        brain_id,
    )
    graph.add_relationship(
        policy,
        _rel("fact-ship-b", "STATES", t1, "help/checkout", "chunk-checkout"),
        ship,
        brain_id,
    )
    graph.add_relationship(
        policy,
        _rel("fact-policy-full", "HAS_POLICY", t1, "help/policy-a", "chunk-policy-a"),
        full,
        brain_id,
    )
    graph.add_relationship(
        policy,
        _rel("fact-policy-credit", "HAS_POLICY", t1, "help/policy-b", "chunk-policy-b"),
        credit,
        brain_id,
    )


@pytest.mark.skipif(not _postgres_ready(), reason="PostgreSQL data backend not available")
def test_help_center_conflict_precision_recall_and_api():
    from src.adapters.data import DataAdapter
    from src.adapters.graph import GraphAdapter
    from src.core.saving.conflict_report import build_report
    from src.core.saving.conflicts import (
        dismiss_conflict,
        get_conflict,
        list_conflicts,
        resolve_conflict,
    )
    from src.lib.postgresql.data import PostgreSQLDataClient
    from src.lib.postgresql.networkx_client import NetworkXGraphClient

    try:
        from src.services.api.controllers import conflicts as conflicts_mod
    except ImportError:
        conflicts_mod = None

    uri = os.getenv("PROGRESSIVE_POSTGRES_TEST_URI")
    original = _configure_postgres(uri)
    brain_id = "cflt" + uuid4().hex[:10]
    data = DataAdapter()
    data.add_client(PostgreSQLDataClient())
    graph = GraphAdapter()
    graph.add_client(NetworkXGraphClient())
    try:
        _seed(data, graph, brain_id)
        found = detect_conflicts(data, graph, brain_id, use_llm=False)
        pairs = {frozenset(item.fact_ids()) for item in found}
        true_pos = pairs & EXPECTED_PAIRS
        false_pos = pairs - EXPECTED_PAIRS
        false_neg = EXPECTED_PAIRS - pairs
        precision = len(true_pos) / len(pairs) if pairs else 0.0
        recall = len(true_pos) / len(EXPECTED_PAIRS)
        assert false_pos == set(), f"false positives: {false_pos}"
        assert false_neg == set(), f"false negatives: {false_neg}"
        assert precision == 1.0
        assert recall == 1.0
        assert not (pairs & FORBIDDEN_PAIRS)

        numeric = next(
            item for item in found if item.disagreement_type == "numeric_mismatch"
        )
        sources = {side.source.source_key for side in numeric.sides}
        assert "help/faq" in sources
        assert any(side.source.chunk_ids for side in numeric.sides)
        assert any(side.source.timestamp for side in numeric.sides)
        assert any(side.source.text for side in numeric.sides)

        other = next(
            item
            for item in found
            if item.id != numeric.id
            and item.disagreement_type != "numeric_mismatch"
        )
        mapping = open_conflicts_by_fact_ids(data, brain_id, ["fact-refund-14"])
        assert mapping["fact-refund-14"]
        empty = open_conflicts_by_fact_ids(data, brain_id, ["unrelated-fact"])
        assert empty["unrelated-fact"] == []
        try:
            from src.services.api.constants.requests import (
                GetContextRequestBody,
                GetContextTriple,
            )
            from src.services.api.controllers.retrieve import _annotate_open_conflicts

            subject = Node(uuid="policy-1", name="RefundPolicy", labels=["POLICY"])
            hub = Node(uuid="hub-1", name="hub", labels=["EVENT"])
            obj = Node(uuid="win-14", name="14 days", labels=["DURATION"])
            fact = Predicate(
                uuid="fact-refund-14", name="HAS_WINDOW", description="", properties={}
            )
            bridge = Predicate(uuid="hub", name="MADE", description="", properties={})
            annotated, lines = _annotate_open_conflicts(
                data,
                brain_id,
                [
                    GetContextTriple(
                        identified_entity="RefundPolicy",
                        triple=(subject, fact, hub, bridge, obj),
                    )
                ],
                ["RefundPolicy HAS_WINDOW 14 days"],
            )
            assert annotated[0].open_conflicts
            assert lines[0].startswith("[open conflict]")
            assert GetContextRequestBody(text="refund window").annotate_conflicts is False
        except ImportError:
            pass

        winning = next(
            side.fact_id
            for side in numeric.sides
            if side.fact_id != "fact-refund-14"
        )
        if conflicts_mod is not None:
            async def _api():
                with patch.object(conflicts_mod, "data_adapter", data):
                    listed = await conflicts_mod.get_conflicts_list(
                        limit=50, skip=0, status="open", brain_id=brain_id
                    )
                    listed_body = json.loads(listed.body.decode("utf-8"))
                    one = await conflicts_mod.get_conflict_by_id(numeric.id, brain_id)
                    one_body = json.loads(one.body.decode("utf-8"))
                    report = await conflicts_mod.get_conflict_report(
                        brain_id=brain_id, fmt="json"
                    )
                    report_body = json.loads(report.body.decode("utf-8"))
                    md = await conflicts_mod.get_conflict_report(
                        brain_id=brain_id, fmt="markdown"
                    )
                    filtered = await conflicts_mod.get_conflicts_list(
                        limit=50,
                        skip=0,
                        source="help/faq",
                        subject="RefundPolicy",
                        brain_id=brain_id,
                    )
                    filtered_body = json.loads(filtered.body.decode("utf-8"))
                    resolved = await conflicts_mod.resolve_conflict_by_id(
                        numeric.id,
                        brain_id,
                        winning_fact_id=winning,
                        method="verified_side",
                    )
                    resolved_body = json.loads(resolved.body.decode("utf-8"))
                    dismissed = await conflicts_mod.dismiss_conflict_by_id(
                        other.id, brain_id, notes="duplicate wording"
                    )
                    dismissed_body = json.loads(dismissed.body.decode("utf-8"))
                    return (
                        listed_body,
                        one_body,
                        report_body,
                        md.body.decode("utf-8"),
                        filtered_body,
                        resolved_body,
                        dismissed_body,
                    )

            (
                listed_body,
                one_body,
                report_body,
                markdown,
                filtered_body,
                resolved_body,
                dismissed_body,
            ) = asyncio.run(_api())
            assert listed_body["total"] >= 4
            assert one_body["conflict"]["id"] == numeric.id
            assert len(one_body["conflict"]["sides"]) == 2
            assert report_body["summary"]["open"] >= 1
            assert "Conflict report" in markdown
            assert filtered_body["count"] >= 1
            assert resolved_body["conflict"]["status"] == "resolved"
            assert resolved_body["conflict"]["resolution"]["method"] == "verified_side"
            assert dismissed_body["conflict"]["status"] == "dismissed"
        else:
            rows, total = list_conflicts(
                data, brain_id, status="open", limit=50, skip=0
            )
            assert total >= 4
            loaded = get_conflict(data, brain_id, numeric.id)
            assert loaded is not None
            assert len(loaded.sides) == 2
            report = build_report(data, brain_id, fmt="markdown")
            assert report["summary"]["open"] >= 1
            assert "Conflict report" in report["markdown"]
            filtered, filtered_total = list_conflicts(
                data,
                brain_id,
                source="help/faq",
                subject="RefundPolicy",
                limit=50,
                skip=0,
            )
            assert filtered_total >= 1
            resolved = resolve_conflict(
                data,
                brain_id,
                numeric.id,
                winning_fact_id=winning,
                method="verified_side",
            )
            assert resolved.status == "resolved"
            assert resolved.resolution.method == "verified_side"
            dismissed = dismiss_conflict(
                data, brain_id, other.id, notes="duplicate wording"
            )
            assert dismissed.status == "dismissed"

        again = detect_conflicts(data, graph, brain_id, use_llm=False)
        kept = {item.id: item.status for item in again}
        assert kept[numeric.id] == "resolved"
        assert kept[other.id] == "dismissed"

        with patch.dict("os.environ", {"CONFLICT_DETECT_ON_INGEST": "false"}):
            maybe_detect_after_ingest(brain_id, ["fact-refund-14"])
    finally:
        _drop_brain(uri, brain_id)
        _restore_postgres(original)
