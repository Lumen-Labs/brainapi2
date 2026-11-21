"""Conflict scans against production Neo4j grouping.

Skipped locally when Neo4j is unreachable. Required in Quality CI
(NEO4J_E2E_REQUIRED / CI). Community Neo4j only has the `neo4j` database.
"""

from __future__ import annotations

import os
import time
from uuid import uuid4

import pytest

from src.constants.kg import Node, Predicate
from src.core.saving.conflicts import (
    _load_triples,
    claim_key_for,
    detect_conflicts,
    normalize_predicate,
    snapshot_from_triple,
)
from src.lib.neo4j.client import normalized_predicate_cypher
from test_conflicts import MemoryConflictStore, PREDICATE_NORMALIZER_CASES

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


@pytest.fixture
def neo4j_conflict_client():
    _require_neo4j()
    from src.lib.neo4j.client import Neo4jClient

    client = Neo4jClient()
    tracked: list[str] = []
    try:
        yield client, tracked
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


def _node(uuid: str, name: str) -> Node:
    return Node(uuid=uuid, name=name, labels=["ENTITY"], properties={})


def _pred(uuid: str, name: str, value: str) -> Predicate:
    return Predicate(
        uuid=uuid,
        name=name,
        description="",
        properties={
            "valid_from": "2024-01-01T00:00:00Z",
            "source_key": uuid,
        },
    )


def _quote_raw_rel_type(name: str) -> str:
    return "`" + name.replace("`", "``") + "`"


def _create_raw_relationship(
    client, subject_uuid: str, rel_type: str, rel_uuid: str, object_uuid: str
) -> None:
    client._run_query(
        "MATCH (a) WHERE a['uuid'] = $subject_uuid\n"
        "MATCH (b) WHERE b['uuid'] = $object_uuid\n"
        "MERGE (a)-[r:"
        + _quote_raw_rel_type(rel_type)
        + " {uuid: $rel_uuid}]->(b)\n"
        "SET r['valid_from'] = $valid_from, r['source_key'] = $rel_uuid",
        BRAIN_ID,
        {
            "subject_uuid": subject_uuid,
            "object_uuid": object_uuid,
            "rel_uuid": rel_uuid,
            "valid_from": "2024-01-01T00:00:00Z",
        },
    )


def test_neo4j_alias_predicates_that_straddle_the_cap_stay_in_one_group(
    neo4j_conflict_client,
):
    client, tracked = neo4j_conflict_client
    prefix = "cscan" + uuid4().hex[:10]
    subject_uuid = f"{prefix}-s"
    objects = {
        "HAS_A": (f"{prefix}-a1", f"{prefix}-oa1", "14 days"),
        "HAS_B": (f"{prefix}-b1", f"{prefix}-ob1", "7 days"),
        "HAS__A": (f"{prefix}-a2", f"{prefix}-oa2", "30 days"),
    }
    tracked.extend([subject_uuid, *(item[1] for item in objects.values())])

    subject = client.add_nodes([_node(subject_uuid, "Subject")], BRAIN_ID)[0]
    created = {}
    for rel_name, (fact_id, object_uuid, value) in objects.items():
        obj = client.add_nodes([_node(object_uuid, value)], BRAIN_ID)[0]
        client.add_relationship(subject, _pred(fact_id, rel_name, value), obj, BRAIN_ID)
        created[rel_name] = fact_id

    listed, _total = client.list_validity_relationships(
        BRAIN_ID, skip=0, limit=10, subject_uuids=[subject_uuid]
    )
    names = [item[1].name for item in listed]
    assert names[:2] == ["HAS_A", "HAS__A"], names
    assert names == ["HAS_A", "HAS__A", "HAS_B"], names
    assert [normalize_predicate(name) for name in names[:2]] == ["HAS_A", "HAS_A"]

    result = detect_conflicts(
        MemoryConflictStore(),
        client,
        BRAIN_ID,
        subject_uuids=[subject_uuid],
        max_relationships=2,
    )
    assert result.scanned_relationships == 2
    assert set(result.scanned_claim_keys) == {claim_key_for(subject_uuid, "HAS_A")}
    assert result.truncated
    assert result.scan_cursor["after_predicate"] == "HAS_A"
    pair_ids = {fact_id for conflict in result for fact_id in conflict.fact_ids()}
    assert created["HAS_A"] in pair_ids
    assert created["HAS__A"] in pair_ids
    assert created["HAS_B"] not in pair_ids


def test_neo4j_pred_key_matches_normalize_predicate(neo4j_conflict_client):
    client, _tracked = neo4j_conflict_client
    query = "RETURN " + normalized_predicate_cypher("$raw") + " AS pred_key"
    result = client._run_query(
        query, BRAIN_ID, {"raw": None}, ensure=False
    )
    assert result.records[0].get("pred_key") == ""
    for raw, expected in PREDICATE_NORMALIZER_CASES:
        result = client._run_query(
            query, BRAIN_ID, {"raw": raw}, ensure=False
        )
        got = result.records[0].get("pred_key")
        assert got == expected, (raw, got, expected)
        assert got == normalize_predicate(raw), (raw, got, normalize_predicate(raw))


def test_neo4j_mixed_raw_types_stay_in_one_group_across_rotation(
    neo4j_conflict_client,
):
    client, tracked = neo4j_conflict_client
    prefix = "cscan" + uuid4().hex[:10]
    subject_uuid = f"{prefix}-s"
    mixed = [
        ("HAS-A", f"{prefix}-a1", f"{prefix}-oa1", "14 days"),
        ("HAS A", f"{prefix}-a2", f"{prefix}-oa2", "30 days"),
        ("HAS.A", f"{prefix}-a3", f"{prefix}-oa3", "21 days"),
        ("has_a", f"{prefix}-a4", f"{prefix}-oa4", "10 days"),
    ]
    other = ("HAS_B", f"{prefix}-b1", f"{prefix}-ob1", "7 days")
    other2 = ("HAS_B", f"{prefix}-b2", f"{prefix}-ob2", "3 days")
    tracked.extend(
        [subject_uuid, *(item[2] for item in mixed), other[2], other2[2]]
    )

    subject = client.add_nodes([_node(subject_uuid, "Subject")], BRAIN_ID)[0]
    for rel_name, fact_id, object_uuid, value in mixed:
        client.add_nodes([_node(object_uuid, value)], BRAIN_ID)
        _create_raw_relationship(
            client, subject_uuid, rel_name, fact_id, object_uuid
        )
    for rel_name, fact_id, object_uuid, value in (other, other2):
        obj = client.add_nodes([_node(object_uuid, value)], BRAIN_ID)[0]
        client.add_relationship(
            subject, _pred(fact_id, rel_name, value), obj, BRAIN_ID
        )

    listed, _total = client.list_validity_relationships(
        BRAIN_ID, skip=0, limit=20, subject_uuids=[subject_uuid]
    )
    names = [item[1].name for item in listed]
    assert len(names) == 6, names
    assert {normalize_predicate(name) for name in names[:4]} == {"HAS_A"}, names
    assert {normalize_predicate(name) for name in names[4:]} == {"HAS_B"}, names

    grouped = detect_conflicts(
        MemoryConflictStore(),
        client,
        BRAIN_ID,
        subject_uuids=[subject_uuid],
        max_relationships=4,
    )
    assert grouped.scanned_relationships == 4
    assert set(grouped.scanned_claim_keys) == {claim_key_for(subject_uuid, "HAS_A")}
    assert grouped.truncated
    assert grouped.scan_cursor["after_predicate"] == "HAS_A"

    first, first_stats = _load_triples(
        client,
        BRAIN_ID,
        subject_uuids=[subject_uuid],
        max_relationships=4,
        rotate=True,
    )
    first_ids = {snapshot_from_triple(item)["fact_id"] for item in first}
    assert first_ids == {row[1] for row in mixed}
    assert first_stats["truncated"]
    cursor = first_stats["scan_cursor"]
    assert cursor["after_predicate"] == "HAS_A"

    second, _second_stats = _load_triples(
        client,
        BRAIN_ID,
        subject_uuids=[subject_uuid],
        max_relationships=4,
        after_subject_uuid=cursor["after_subject"],
        after_predicate=cursor["after_predicate"],
        rotate=True,
    )
    second_ids = {snapshot_from_triple(item)["fact_id"] for item in second}
    assert second_ids == {other[1], other2[1]}
    assert first_ids.isdisjoint(second_ids)
    assert first_ids | second_ids == {item[1].uuid for item in listed}
