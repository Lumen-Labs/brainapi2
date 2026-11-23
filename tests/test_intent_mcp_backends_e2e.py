"""Intent MCP recall/remember against real postgres / neo4j backends.

Skipped unless the Quality CI services are present:
``PROGRESSIVE_POSTGRES_TEST_URI`` for postgres, ``NEO4J_E2E_REQUIRED`` plus a
reachable bolt server for neo4j. No LLM is invoked.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, wait
from uuid import uuid4

import pytest
from unittest.mock import patch

from src.core.intent.service import check_claim, recall_facts, remember_fact
from src.core.saving.validity import VALID_FROM, VALID_TO


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
    from urllib.parse import urlparse

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


def _neo4j_required() -> bool:
    flag = os.getenv("NEO4J_E2E_REQUIRED", "")
    return flag.lower() in {"1", "true", "yes"} or os.getenv("CI") == "true"


def _neo4j_connectable() -> bool:
    from src.config import config

    config.neo4j.host = os.getenv("NEO4J_HOST", config.neo4j.host)
    config.neo4j.port = os.getenv("NEO4J_PORT", config.neo4j.port)
    config.neo4j.username = os.getenv("NEO4J_USERNAME", config.neo4j.username)
    config.neo4j.password = os.getenv("NEO4J_PASSWORD", config.neo4j.password)
    try:
        from neo4j import GraphDatabase

        driver = GraphDatabase.driver(
            f"bolt://{config.neo4j.host}:{config.neo4j.port}",
            auth=(config.neo4j.username, config.neo4j.password),
        )
        driver.verify_connectivity()
        driver.close()
        return True
    except Exception:
        return False


def _validity(graph, fact_id, brain_id):
    row = graph.get_fact_record(fact_id, brain_id)
    props = (row or {}).get("properties") or {}
    return props.get(VALID_FROM) or props.get("valid_at"), props.get(VALID_TO) or props.get(
        "invalid_at"
    )


@pytest.fixture
def no_llm():
    with patch("src.core.intent.parse._configured_llm", return_value=None):
        yield


@pytest.mark.skipif(not _postgres_ready(), reason="PostgreSQL data backend not available")
def test_postgres_intent_recall_check_and_validity_order(no_llm):
    from src.adapters.data import DataAdapter
    from src.adapters.graph import GraphAdapter
    from src.lib.postgresql.data import PostgreSQLDataClient
    from src.lib.postgresql.networkx_client import NetworkXGraphClient

    uri = os.getenv("PROGRESSIVE_POSTGRES_TEST_URI")
    original = _configure_postgres(uri)
    brain_id = "intent" + uuid4().hex[:10]
    data = DataAdapter()
    data.add_client(PostgreSQLDataClient())
    graph = GraphAdapter()
    graph.add_client(NetworkXGraphClient())
    try:
        newer = remember_fact(
            fact="Alice lives in Milan",
            source="handbook.md",
            brain_id=brain_id,
            token="pat-alpha",
            valid_from="2024-01-01T00:00:00Z",
            graph=graph,
            data=data,
        )
        older = remember_fact(
            fact="Alice lives in Rome",
            source="handbook.md",
            brain_id=brain_id,
            token="pat-alpha",
            valid_from="2020-01-01T00:00:00Z",
            graph=graph,
            data=data,
        )
        older_from, older_to = _validity(graph, older.fact.fact_id, brain_id)
        newer_from, newer_to = _validity(graph, newer.fact.fact_id, brain_id)
        assert older_from == "2020-01-01T00:00:00Z"
        assert older_to == "2024-01-01T00:00:00Z"
        assert older_to >= older_from
        assert not newer_to
        assert newer_from == "2024-01-01T00:00:00Z"

        live = recall_facts(
            question="where does Alice live",
            brain_id=brain_id,
            graph=graph,
            data=data,
        )
        assert live.matched
        assert live.facts[0].object == "Milan"

        historical = recall_facts(
            question="where does Alice live",
            brain_id=brain_id,
            as_of="2021-06-01T00:00:00Z",
            graph=graph,
            data=data,
        )
        assert historical.facts[0].object == "Rome"

        empty = recall_facts(
            question="quantum chromodynamics lattice spacing",
            brain_id=brain_id,
            graph=graph,
            data=data,
        )
        assert empty.facts == []
        assert empty.matched is False

        unknown = check_claim(
            claim="Refund window is 30 days",
            brain_id=brain_id,
            graph=graph,
            data=data,
        )
        assert unknown.verdict == "unknown"
        assert unknown.evidence == []
    finally:
        _drop_brain(uri, brain_id)
        _restore_postgres(original)


@pytest.mark.skipif(not _postgres_ready(), reason="PostgreSQL data backend not available")
def test_postgres_concurrent_remember_leaves_one_current(no_llm):
    from src.adapters.data import DataAdapter
    from src.adapters.graph import GraphAdapter
    from src.lib.postgresql.data import PostgreSQLDataClient
    from src.lib.postgresql.networkx_client import NetworkXGraphClient

    uri = os.getenv("PROGRESSIVE_POSTGRES_TEST_URI")
    original = _configure_postgres(uri)
    brain_id = "intentc" + uuid4().hex[:10]
    data = DataAdapter()
    data.add_client(PostgreSQLDataClient())
    graph = GraphAdapter()
    graph.add_client(NetworkXGraphClient())

    def write(city):
        return remember_fact(
            fact=f"Alice lives in {city}",
            source="handbook.md",
            brain_id=brain_id,
            token="pat-alpha",
            valid_from="2024-06-01T00:00:00Z",
            graph=graph,
            data=data,
        )

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(write, city) for city in ("Rome", "Milan")]
            wait(futures)
            for future in futures:
                future.result()
        live = recall_facts(
            question="where does Alice live",
            brain_id=brain_id,
            graph=graph,
            data=data,
        )
        current = [fact for fact in live.facts if fact.status in {"verified", "unverified"}]
        assert len(current) == 1
        for fact in live.facts:
            if fact.valid_at and fact.invalid_at:
                assert fact.invalid_at >= fact.valid_at
    finally:
        _drop_brain(uri, brain_id)
        _restore_postgres(original)


def test_neo4j_intent_recall_and_backdating():
    if not _neo4j_connectable():
        if _neo4j_required():
            pytest.fail("Neo4j e2e is required but Neo4j is not reachable")
        pytest.skip("Neo4j is not available")
    uri = os.getenv("PROGRESSIVE_POSTGRES_TEST_URI")
    if not uri:
        if _neo4j_required():
            pytest.fail("Neo4j intent e2e requires PROGRESSIVE_POSTGRES_TEST_URI")
        pytest.skip("PostgreSQL data backend not available")

    from src.adapters.data import DataAdapter
    from src.adapters.graph import GraphAdapter
    from src.lib.neo4j.client import Neo4jClient
    from src.lib.postgresql.data import PostgreSQLDataClient

    original = _configure_postgres(uri)
    brain_id = "neo4j"
    person = "IntentAlice" + uuid4().hex[:8]
    client = Neo4jClient()
    graph = GraphAdapter()
    graph.add_client(client)
    data = DataAdapter()
    data.add_client(PostgreSQLDataClient())
    tracked: list[str] = []
    try:
        with patch("src.core.intent.parse._configured_llm", return_value=None):
            newer = remember_fact(
                fact=f"{person} lives in Milan",
                source="handbook.md",
                brain_id=brain_id,
                token="pat-alpha",
                valid_from="2024-01-01T00:00:00Z",
                graph=graph,
                data=data,
            )
            older = remember_fact(
                fact=f"{person} lives in Rome",
                source="handbook.md",
                brain_id=brain_id,
                token="pat-alpha",
                valid_from="2020-01-01T00:00:00Z",
                graph=graph,
                data=data,
            )
        if newer.fact:
            tracked.append(newer.fact.fact_id)
        if older.fact:
            tracked.append(older.fact.fact_id)
        _from, until = _validity(graph, older.fact.fact_id, brain_id)
        assert until == "2024-01-01T00:00:00Z"
        assert until >= _from
        live = recall_facts(
            question=f"where does {person} live",
            brain_id=brain_id,
            graph=graph,
            data=data,
        )
        assert live.matched
        assert live.facts[0].object == "Milan"
    finally:
        if tracked:
            try:
                client._run_query(
                    "MATCH ()-[r]->() WHERE r['uuid'] IN $uuids DELETE r",
                    brain_id,
                    {"uuids": tracked},
                )
            except Exception:
                pass
        try:
            client.driver.close()
        except Exception:
            pass
        _drop_brain(uri, brain_id)
        _restore_postgres(original)
