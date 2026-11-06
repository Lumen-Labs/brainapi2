"""Per-brain snapshots and bounded SQL reads. No request loads the complete graph."""

import base64
import hashlib
import json
import logging
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime
from functools import lru_cache

from psycopg2.extras import Json, execute_values

from src.lib.postgresql.graph_store import PostgreSQLGraphStore

from .communities import build_communities

logger = logging.getLogger(__name__)
DDL = """
CREATE TABLE IF NOT EXISTS explore_snapshots (
 id TEXT PRIMARY KEY, created_at TIMESTAMPTZ NOT NULL DEFAULT now(), payload JSONB NOT NULL
);
CREATE TABLE IF NOT EXISTS explore_memberships (
 snapshot_id TEXT NOT NULL REFERENCES explore_snapshots(id) ON DELETE CASCADE,
 community_id TEXT NOT NULL, entity_id TEXT NOT NULL,
 PRIMARY KEY(snapshot_id, community_id, entity_id)
);
CREATE INDEX IF NOT EXISTS explore_membership_entity ON explore_memberships(snapshot_id, entity_id);
CREATE TABLE IF NOT EXISTS explore_state (
 singleton BOOLEAN PRIMARY KEY DEFAULT TRUE CHECK(singleton), active_snapshot TEXT,
 build_token TEXT, build_until TIMESTAMPTZ, failed BOOLEAN NOT NULL DEFAULT FALSE
);
INSERT INTO explore_state(singleton) VALUES(TRUE) ON CONFLICT DO NOTHING;
CREATE INDEX IF NOT EXISTS idx_kg_relationships_target ON kg_relationships(target_uuid);
"""


class SnapshotExpired(Exception):
    pass


class InvalidCursor(Exception):
    pass


def encode_cursor(snapshot, scope, offsets, filters):
    value = {
        "s": snapshot,
        "g": scope,
        "o": offsets,
        "f": hashlib.sha256(json.dumps(filters, sort_keys=True).encode()).hexdigest(),
    }
    return base64.urlsafe_b64encode(
        json.dumps(value, separators=(",", ":")).encode()
    ).decode()


def decode_cursor(value, snapshot, scope, filters):
    if not value:
        return ["", ""]
    try:
        if len(value) > 2048:
            raise ValueError()
        data = json.loads(base64.urlsafe_b64decode(value))
        expected = hashlib.sha256(
            json.dumps(filters, sort_keys=True).encode()
        ).hexdigest()
        if data["s"] != snapshot:
            raise SnapshotExpired()
        if data["g"] != scope or data["f"] != expected:
            raise ValueError()
        offsets = data["o"]
        if len(offsets) != 2 or any(
            not isinstance(n, str) or len(n) > 512 for n in offsets
        ):
            raise ValueError()
        return offsets
    except SnapshotExpired:
        raise
    except (ValueError, KeyError, TypeError):
        raise InvalidCursor() from None


def labels(value):
    return (
        [item for item in value if isinstance(item, str) and item]
        if isinstance(value, list)
        else []
    )


def details(data):
    system = {
        "uuid",
        "name",
        "labels",
        "description",
        "last_updated",
        "metadata",
        "observations",
        "v_id",
        "direction",
    }

    def scalar(value):
        return (
            value
            if value is None or isinstance(value, (str, int, float, bool))
            else json.dumps(value, ensure_ascii=False)
        )

    return {
        key: [scalar(v) for v in value] if isinstance(value, list) else scalar(value)
        for key, value in data.items()
        if key not in system
    }


def thing(row):
    node_id, data = row
    types = labels(data.get("labels"))
    return {
        "id": node_id,
        "name": str(data.get("name") or "Unnamed"),
        "types": types,
        "typeLabels": [t.replace("_", " ").title() for t in types],
        "description": data.get("description")
        if isinstance(data.get("description"), str)
        else None,
        "details": details(data),
        "updatedAt": data.get("last_updated")
        if isinstance(data.get("last_updated"), str)
        else None,
        "advanced": {},
    }


def connection(row):
    edge_id, source, target, rel_type, data = row
    return {
        "id": edge_id,
        "fromId": source,
        "toId": target,
        "type": rel_type,
        "typeLabel": rel_type.replace("_", " ").title(),
        "direction": data.get("direction")
        if data.get("direction") in ("in", "out", "neutral")
        else "out",
        "description": data.get("description")
        if isinstance(data.get("description"), str)
        else None,
        "details": details(data),
        "updatedAt": data.get("last_updated")
        if isinstance(data.get("last_updated"), str)
        else None,
        "advanced": {},
    }


class AtlasStore:
    def __init__(self):
        self.graph = PostgreSQLGraphStore()

    @contextmanager
    def transaction(self, brain_id):
        with self.graph._connection(brain_id) as conn:
            try:
                with conn.cursor() as cur:
                    # DDL is idempotent and guarded across API/worker processes.
                    cur.execute("SELECT pg_advisory_xact_lock(19650614)")
                    cur.execute(DDL)
                conn.commit()
                with conn.cursor() as cur:
                    cur.execute("SET LOCAL statement_timeout = '10000ms'")
                    yield cur
                conn.commit()
            except Exception:
                conn.rollback()
                raise

    def overview(self, brain_id):
        token = None
        with self.transaction(brain_id) as cur:
            cur.execute("""SELECT s.id, s.created_at, s.payload, st.build_until > now(), st.failed
                FROM explore_state st LEFT JOIN explore_snapshots s ON s.id=st.active_snapshot WHERE st.singleton""")
            snapshot_id, created, payload, building, failed = cur.fetchone()
            stale = (
                created is None or (datetime.now(UTC) - created).total_seconds() > 300
            )
            if stale and not building:
                token = uuid.uuid4().hex
                cur.execute(
                    """UPDATE explore_state SET build_token=%s, build_until=now()+interval '15 minutes', failed=FALSE
                    WHERE singleton AND (build_until IS NULL OR build_until < now()) RETURNING singleton""",
                    (token,),
                )
                if not cur.fetchone():
                    token = None
        if token:
            try:
                from src.workers.tasks.explore import build_atlas

                build_atlas.apply_async(
                    args=[brain_id, token], queue="consolidate_graph"
                )
                building, failed = True, False
            except Exception:
                self.fail(brain_id, token)
                building, failed = False, True
                logger.exception("explore_overview_enqueue_failed")
        return snapshot_id, created, payload, bool(building), bool(failed)

    def fail(self, brain_id, token):
        with self.transaction(brain_id) as cur:
            cur.execute(
                "UPDATE explore_state SET build_until=NULL, build_token=NULL, failed=TRUE WHERE singleton AND build_token=%s",
                (token,),
            )

    def build(self, brain_id, token):
        import time

        started = time.monotonic()
        try:
            with self.transaction(brain_id) as cur:
                # Snapshot all input in one consistent database transaction.
                cur.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ")
                cur.execute("SELECT build_token FROM explore_state WHERE singleton")
                if cur.fetchone()[0] != token:
                    return
                cur.execute(
                    "SELECT uuid, data->>'name', data->'labels' FROM kg_nodes ORDER BY uuid"
                )
                nodes = [
                    {"id": n, "name": name or "Unnamed", "types": types or []}
                    for n, name, types in cur.fetchall()
                ]
                cur.execute(
                    "SELECT source_uuid, target_uuid FROM kg_relationships ORDER BY uuid"
                )
                edges = [{"fromId": s, "toId": t} for s, t in cur.fetchall()]
            payload, memberships = build_communities(nodes, edges)
            snapshot_id = uuid.uuid4().hex
            with self.transaction(brain_id) as cur:
                cur.execute(
                    "SELECT build_token FROM explore_state WHERE singleton FOR UPDATE"
                )
                if cur.fetchone()[0] != token:
                    return
                cur.execute(
                    "INSERT INTO explore_snapshots(id,payload) VALUES(%s,%s)",
                    (snapshot_id, Json(payload)),
                )
                if memberships:
                    execute_values(
                        cur,
                        "INSERT INTO explore_memberships(snapshot_id,community_id,entity_id) VALUES %s",
                        [(snapshot_id, group, entity) for group, entity in memberships],
                        page_size=2000,
                    )
                cur.execute(
                    "UPDATE explore_state SET active_snapshot=%s,build_token=NULL,build_until=NULL,failed=FALSE WHERE singleton",
                    (snapshot_id,),
                )
                cur.execute(
                    "DELETE FROM explore_snapshots WHERE id<>%s AND created_at < now()-interval '1 hour'",
                    (snapshot_id,),
                )
            logger.info(
                "explore_overview_built duration_ms=%d nodes=%d edges=%d",
                int((time.monotonic() - started) * 1000),
                len(nodes),
                len(edges),
            )
        except Exception:
            self.fail(brain_id, token)
            logger.exception("explore_overview_build_failed")
            raise

    def read(
        self,
        brain_id,
        group=None,
        snapshot=None,
        node=None,
        types=(),
        connection_types=(),
        cursor=None,
    ):
        if node:
            snapshot_id, created, payload, building, failed = (
                None,
                None,
                None,
                False,
                False,
            )
        elif group or snapshot:
            with self.transaction(brain_id) as cur:
                cur.execute(
                    "SELECT id,created_at,payload FROM explore_snapshots WHERE id=%s",
                    (snapshot,),
                )
                row = cur.fetchone()
                if not row:
                    raise SnapshotExpired()
                snapshot_id, created, payload = row
            building, failed = False, False
        else:
            snapshot_id, created, payload, building, failed = self.overview(brain_id)
        base = {
            "version": 1,
            "snapshotId": snapshot_id,
            "builtAt": created.isoformat() if created else None,
            "status": "refreshing"
            if building and payload
            else "building"
            if building
            else "failed"
            if failed
            else "ready",
            "groups": [],
            "groupLinks": [],
            "things": [],
            "connections": [],
            "nextCursor": None,
            "totals": (payload or {}).get(
                "totals", {"things": None, "connections": None}
            ),
            "ancestors": [],
        }
        if not node and not payload:
            return base
        all_groups = (payload or {}).get("groups", [])
        selected = next((g for g in all_groups if g["id"] == group), None)
        if group and not selected:
            raise SnapshotExpired()
        parent = selected
        while parent:
            base["ancestors"].insert(0, {"id": parent["id"], "name": parent["name"]})
            parent = next(
                (g for g in all_groups if g["id"] == parent["parentId"]), None
            )
        children = [g for g in all_groups if g["parentId"] == group] if not node else []
        filters = [sorted(types), sorted(connection_types)]
        scope = node or group or "overview"
        offsets = decode_cursor(cursor, snapshot_id, scope, filters)
        with self.transaction(brain_id) as cur:
            # Endpoint filters use both ends consistently, including neighborhood reads.
            eligible = "SELECT uuid FROM kg_nodes WHERE (%s::text[]='{}' OR COALESCE(data->'labels','[]'::jsonb) ?| %s::text[])"
            eligible_args = [list(types), list(types)]
            if children:
                ids = [g["id"] for g in children]
                cur.execute(
                    f"""WITH eligible AS ({eligible}) SELECT m.community_id,COUNT(*) FROM explore_memberships m
                    JOIN eligible e ON e.uuid=m.entity_id WHERE m.snapshot_id=%s AND m.community_id=ANY(%s)
                    GROUP BY m.community_id""",
                    (*eligible_args, snapshot_id, ids),
                )
                counts = dict(cur.fetchall())
                base["groups"] = [
                    {**g, "count": counts[g["id"]]}
                    for g in children
                    if counts.get(g["id"])
                ]
                cur.execute(
                    f"""WITH eligible AS ({eligible}) SELECT a.community_id,b.community_id,COUNT(*)
                    FROM kg_relationships r JOIN eligible es ON es.uuid=r.source_uuid JOIN eligible et ON et.uuid=r.target_uuid
                    JOIN explore_memberships a ON a.entity_id=r.source_uuid AND a.snapshot_id=%s AND a.community_id=ANY(%s)
                    JOIN explore_memberships b ON b.entity_id=r.target_uuid AND b.snapshot_id=%s AND b.community_id=ANY(%s)
                    WHERE a.community_id<>b.community_id AND (%s::text[]='{{}}' OR r.rel_type=ANY(%s))
                    GROUP BY a.community_id,b.community_id ORDER BY COUNT(*) DESC, a.community_id,b.community_id LIMIT 4001""",
                    (
                        *eligible_args,
                        snapshot_id,
                        ids,
                        snapshot_id,
                        ids,
                        list(connection_types),
                        list(connection_types),
                    ),
                )
                link_rows = cur.fetchall()
                base["groupLinksTruncated"] = len(link_rows) > 4000
                base["groupLinks"] = [
                    {"id": f"aggregate:{a}:{b}", "fromId": a, "toId": b, "count": count}
                    for a, b, count in link_rows[:4000]
                ]
                base["coverage"] = {
                    "things": sum(counts.values()),
                    "connections": sum(e["count"] for e in base["groupLinks"]),
                    "aggregated": True,
                }
                return base
            if node:
                cur.execute("SELECT uuid,data FROM kg_nodes WHERE uuid=%s", (node,))
                center = cur.fetchone()
                if not center:
                    raise KeyError(node)
                member = eligible
                member_args = eligible_args
                condition = "(r.source_uuid=%s OR r.target_uuid=%s)"
                condition_args = [node, node]
            else:
                member = f"SELECT e.uuid FROM ({eligible}) e JOIN explore_memberships m ON m.entity_id=e.uuid WHERE m.snapshot_id=%s AND m.community_id=%s"
                member_args = [*eligible_args, snapshot_id, group]
                condition, condition_args = "TRUE", []
            edge_from = f"""FROM kg_relationships r JOIN members s ON s.uuid=r.source_uuid JOIN members t ON t.uuid=r.target_uuid
                WHERE {condition} AND (%s::text[]='{{}}' OR r.rel_type=ANY(%s))"""
            args = [
                *member_args,
                *condition_args,
                list(connection_types),
                list(connection_types),
            ]
            cur.execute(f"WITH members AS ({member}) SELECT COUNT(*) {edge_from}", args)
            edge_total = cur.fetchone()[0]
            cur.execute(
                f"WITH members AS ({member}) SELECT r.uuid,r.source_uuid,r.target_uuid,r.rel_type,r.data {edge_from} AND r.uuid > %s ORDER BY r.uuid LIMIT 301",
                (*args, offsets[1]),
            )
            rows = cur.fetchall()
            edges = rows[:300]
            if node:
                node_rows = [center]
                node_total = None
                has_nodes = False
            else:
                cur.execute(
                    f"WITH members AS ({member}) SELECT COUNT(*) FROM members",
                    member_args,
                )
                node_total = cur.fetchone()[0]
                cur.execute(
                    f"WITH members AS ({member}) SELECT n.uuid,n.data FROM kg_nodes n JOIN members m ON m.uuid=n.uuid WHERE n.uuid > %s ORDER BY n.uuid LIMIT 301",
                    (*member_args, offsets[0]),
                )
                page = cur.fetchall()
                node_rows, has_nodes = page[:300], len(page) > 300
            endpoints = sorted({e[1] for e in edges} | {e[2] for e in edges})
            cur.execute(
                "SELECT uuid,data FROM kg_nodes WHERE uuid=ANY(%s)", (endpoints,)
            )
            node_rows = list({r[0]: r for r in [*node_rows, *cur.fetchall()]}.values())
            base["things"], base["connections"] = (
                [thing(r) for r in node_rows],
                [connection(r) for r in edges],
            )
            base["totals"] = {"things": node_total, "connections": edge_total}
            if has_nodes or len(rows) > 300:
                base["nextCursor"] = encode_cursor(
                    snapshot_id,
                    scope,
                    [
                        offsets[0] if node or not page[:300] else page[:300][-1][0],
                        offsets[1] if not edges else edges[-1][0],
                    ],
                    filters,
                )
            base["coverage"] = {
                "things": len(node_rows),
                "connections": len(edges),
                "aggregated": False,
            }
            return base


@lru_cache(maxsize=1)
def atlas_store():
    return AtlasStore()
