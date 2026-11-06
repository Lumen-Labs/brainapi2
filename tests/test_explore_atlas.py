import base64
import json
from unittest.mock import Mock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.services.api.dependencies import get_brain_id
from src.services.api.routes.explore import explore_router
from src.services.explore.communities import build_communities
from src.services.explore.store import (
    InvalidCursor,
    SnapshotExpired,
    decode_cursor,
    encode_cursor,
)


def fixture(n=30):
    nodes = [
        {
            "id": str(i),
            "name": f"Entity {i}",
            "types": ["PERSON" if i % 2 else "PROJECT"],
        }
        for i in range(n)
    ]
    edges = [{"fromId": str(i), "toId": str((i + 1) % 20)} for i in range(20)]
    return nodes, edges


def test_complete_deterministic_membership_with_isolates_and_parallel_edges():
    nodes, edges = fixture()
    edges += [
        edges[0],
        {"fromId": "0", "toId": "0"},
        {"fromId": "missing", "toId": "0"},
    ]
    snapshot, memberships = build_communities(nodes, edges)
    assert build_communities(list(reversed(nodes)), list(reversed(edges))) == (
        snapshot,
        memberships,
    )
    roots = {g["id"] for g in snapshot["groups"] if g["parentId"] is None}
    root_members = [n for group, n in memberships if group in roots]
    assert sorted(root_members) == sorted(n["id"] for n in nodes)
    assert snapshot["totals"] == {"things": 30, "connections": 22}
    {g["id"]: g for g in snapshot["groups"]}
    for group in snapshot["groups"]:
        members = {n for cid, n in memberships if cid == group["id"]}
        assert len(members) == group["count"]
        if group["parentId"]:
            assert members <= {n for cid, n in memberships if cid == group["parentId"]}
        assert group["representatives"]


def test_empty_and_self_loop_only_graphs():
    assert build_communities([], [])[0]["groups"] == []
    snapshot, membership = build_communities(
        [{"id": "a", "name": "A", "types": []}], [{"fromId": "a", "toId": "a"}]
    )
    assert snapshot["totals"] == {"things": 1, "connections": 1}
    assert membership[0][1] == "a"


def test_cursor_is_bound_to_snapshot_scope_and_filters():
    cursor = encode_cursor(
        "snapshot", "group", ["node-300", "edge-600"], [["PERSON"], []]
    )
    assert decode_cursor(cursor, "snapshot", "group", [["PERSON"], []]) == [
        "node-300",
        "edge-600",
    ]
    with pytest.raises(SnapshotExpired):
        decode_cursor(cursor, "other", "group", [["PERSON"], []])
    with pytest.raises(InvalidCursor):
        decode_cursor(cursor, "snapshot", "other", [["PERSON"], []])
    with pytest.raises(InvalidCursor):
        decode_cursor(cursor, "snapshot", "group", [[], []])
    for malformed in [
        "garbage!",
        "x" * 3000,
        base64.urlsafe_b64encode(json.dumps({"o": [-1, 0]}).encode()).decode(),
    ]:
        with pytest.raises(InvalidCursor):
            decode_cursor(malformed, "snapshot", "group", [[], []])


@pytest.fixture
def api():
    app = FastAPI()
    app.include_router(explore_router, prefix="/retrieve")
    app.dependency_overrides[get_brain_id] = lambda: "authorized-brain"
    return TestClient(app)


def test_api_passes_authorized_brain_and_filters(api):
    store = Mock()
    store.read.return_value = {"version": 1, "status": "ready"}
    with (
        patch("src.services.api.routes.explore.config.graph_db", "networkx"),
        patch("src.services.api.routes.explore.atlas_store", return_value=store),
    ):
        response = api.get("/retrieve/explore/v1/overview?types=PERSON,PROJECT")
    assert response.status_code == 200
    store.read.assert_called_once_with(
        "authorized-brain", types=["PERSON", "PROJECT"], connection_types=[]
    )


@pytest.mark.parametrize(
    "error,status",
    [(SnapshotExpired(), 409), (InvalidCursor(), 422), (KeyError("gone"), 404)],
)
def test_api_recoverable_errors(api, error, status):
    with (
        patch("src.services.api.routes.explore.config.graph_db", "networkx"),
        patch("src.services.api.routes.explore.atlas_store") as store,
    ):
        store.return_value.read.side_effect = error
        assert (
            api.get(
                "/retrieve/explore/v1/communities/group:one?snapshot=old"
            ).status_code
            == status
        )


def test_older_graph_backend_has_explicit_fallback(api):
    with patch("src.services.api.routes.explore.config.graph_db", "neo4j"):
        assert api.get("/retrieve/explore/v1/overview").status_code == 501


def test_graph_payload_preserves_inspection_properties_and_normalizes_labels():
    from src.services.explore.store import connection, thing

    entity = thing(
        (
            "a",
            {
                "name": "A",
                "labels": ["PERSON", 42],
                "description": "Description",
                "score": 3,
                "nested": {"value": 1},
                "last_updated": "2026-11-05T12:00:00Z",
            },
        )
    )
    assert entity["types"] == ["PERSON"]
    assert entity["details"]["score"] == 3
    assert json.loads(entity["details"]["nested"]) == {"value": 1}
    assert "name" not in entity["details"]
    assert entity["updatedAt"] == "2026-11-05T12:00:00Z"
    edge = connection(
        ("r", "a", "b", "KNOWS", {"direction": "neutral", "confidence": 0.8})
    )
    assert edge["direction"] == "neutral"
    assert edge["details"] == {"confidence": 0.8}
