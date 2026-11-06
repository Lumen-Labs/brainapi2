"""Run with ATLAS_TEST_POSTGRES=1 against a disposable local database only."""

import os
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock, patch
from uuid import uuid4

import psycopg2
import pytest
from psycopg2 import sql
from psycopg2.extras import Json, execute_values

from src.services.explore.store import AtlasStore, SnapshotExpired

pytestmark = pytest.mark.skipif(
    os.getenv("ATLAS_TEST_POSTGRES") != "1",
    reason="Requires local disposable PostgreSQL database",
)


@pytest.fixture
def store():
    db = "atlas_test_" + uuid4().hex
    # Unix socket, current OS user; never use the application's credentials.
    admin = psycopg2.connect(dbname="postgres")
    admin.autocommit = True
    with admin.cursor() as cur:
        cur.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(db)))
    conn = psycopg2.connect(dbname=db)
    with conn.cursor() as cur:
        cur.execute("CREATE TABLE kg_nodes(uuid TEXT PRIMARY KEY,data JSONB NOT NULL)")
        cur.execute(
            "CREATE TABLE kg_relationships(uuid TEXT PRIMARY KEY,source_uuid TEXT,target_uuid TEXT,rel_type TEXT,data JSONB)"
        )
        execute_values(
            cur,
            "INSERT INTO kg_nodes VALUES %s",
            [
                (
                    str(i),
                    Json(
                        {
                            "name": f"Entity {i}",
                            "labels": ["PERSON" if i % 2 else "PROJECT"],
                        }
                    ),
                )
                for i in range(800)
            ],
        )
        execute_values(
            cur,
            "INSERT INTO kg_relationships VALUES %s",
            [(f"e{i:04}", "0", str(i), "KNOWS", Json({})) for i in range(1, 800)],
        )
    conn.commit()

    @contextmanager
    def connection(brain_id):
        assert brain_id == "test-brain"
        yield conn

    value = object.__new__(AtlasStore)
    value.graph = SimpleNamespace(_connection=connection)
    try:
        yield value
    finally:
        conn.close()
        with admin.cursor() as cur:
            cur.execute(sql.SQL("DROP DATABASE {}").format(sql.Identifier(db)))
        admin.close()


def test_background_snapshot_publish_and_progressive_reads(store):
    task = Mock()
    with patch.dict(
        "sys.modules", {"src.workers.tasks.explore": SimpleNamespace(build_atlas=task)}
    ):
        first = store.read("test-brain")
        assert first["status"] == "building"
        token = task.apply_async.call_args.kwargs["args"][1]
        second = store.read("test-brain")
        assert second["status"] == "building"
        task.apply_async.assert_called_once()
    store.build("test-brain", token)
    overview = store.read("test-brain")
    assert overview["status"] == "ready"
    assert overview["coverage"]["things"] == 800
    assert sum(g["count"] for g in overview["groups"]) == 800
    group = overview["groups"][0]["id"]
    detail = store.read("test-brain", group=group, snapshot=overview["snapshotId"])
    assert detail["things"]
    assert len(detail["connections"]) <= 300
    # High-degree neighborhood pagination must expose every relationship once.
    ids, cursor = set(), None
    for _ in range(10):
        page = store.read("test-brain", node="0", cursor=cursor)
        page_ids = {e["id"] for e in page["connections"]}
        assert not ids & page_ids
        ids |= page_ids
        cursor = page["nextCursor"]
        if not cursor:
            break
    assert len(ids) == 799
    assert cursor is None
    filtered = store.read("test-brain", node="0", connection_types=["MISSING"])
    assert filtered["connections"] == []
    assert [t["id"] for t in filtered["things"]] == ["0"]
    with pytest.raises(SnapshotExpired):
        store.read("test-brain", group=group, snapshot="expired")
    with pytest.raises(AssertionError):
        store.read("other-brain")


def test_failed_build_keeps_previous_snapshot_and_releases_claim(store):
    task = Mock()
    with patch.dict(
        "sys.modules", {"src.workers.tasks.explore": SimpleNamespace(build_atlas=task)}
    ):
        store.overview("test-brain")
    token = task.apply_async.call_args.kwargs["args"][1]
    store.build("test-brain", token)
    with store.transaction("test-brain") as cur:
        cur.execute(
            "UPDATE explore_state SET build_token='new-build',build_until=now()+interval '1 minute'"
        )
    before = store.read("test-brain")["snapshotId"]
    with (
        patch(
            "src.services.explore.store.build_communities",
            side_effect=RuntimeError("failure"),
        ),
        pytest.raises(RuntimeError),
    ):
        store.build("test-brain", "new-build")
    after = store.read("test-brain")
    assert after["snapshotId"] == before
    assert after["status"] == "failed"
