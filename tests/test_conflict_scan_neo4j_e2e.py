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
from src.core.saving.conflicts import claim_key_for, detect_conflicts, normalize_predicate
from test_conflicts import MemoryConflictStore

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
