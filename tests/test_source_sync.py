import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from src.constants.data import TextChunk
from src.constants.tasks.ingestion import IngestionTaskArgs
from src.core.plugins.catalog import PluginManifestCatalog
from src.core.plugins.celery_discovery import discover_plugin_celery

PLUGIN_DIR = Path(__file__).resolve().parent.parent / "plugins" / "source_sync"
if str(PLUGIN_DIR) not in sys.path:
    sys.path.insert(0, str(PLUGIN_DIR))

from source_sync_apply import apply_event, close_citing_facts, dead_letter, event_from_payload
from source_sync_connector import (
    SourceFetchError,
    WebhookRejected,
    ZendeskGuideConnector,
    generic_hmac_hex,
    zendesk_signature,
)
from source_sync_store import SourceSyncStore, public_source
from source_sync_api import create_admin_router, create_webhook_router
from main import register


class FakeData:
    def __init__(self, chunks=None):
        self.chunks = list(chunks or [])
        self.marked = []

    def get_text_chunks(self, brain_id, limit=10, skip=0, query_text=None, metadata_eq=None, order="desc"):
        matched = []
        for chunk in self.chunks:
            meta = chunk.metadata or {}
            if metadata_eq and any(meta.get(key) != value for key, value in metadata_eq.items()):
                continue
            matched.append(chunk)
        return matched[skip : skip + limit], len(matched)

    def mark_text_chunks_superseded(
        self, ids, superseded_at, brain_id, superseded_by=None, valid_to=None
    ):
        self.marked.append((list(ids), superseded_at, superseded_by, valid_to))
        for chunk in self.chunks:
            if str(chunk.id) in ids:
                chunk.metadata["superseded_at"] = superseded_at
                if valid_to:
                    chunk.metadata["valid_to"] = valid_to


class FakeGraph:
    def __init__(self, triples=None):
        self.triples = list(triples or [])
        self.updated = []

    def nodes_and_edges_citing(self, brain_id, chunk_ids, edge_uuids):
        return [], self.triples

    def update_properties(
        self,
        uuid,
        updating,
        brain_id="default",
        new_properties=None,
        properties_to_remove=None,
        expected_open=False,
    ):
        self.updated.append(
            {
                "uuid": uuid,
                "updating": updating,
                "new_properties": dict(new_properties or {}),
                "expected_open": expected_open,
            }
        )


class FakeContext:
    def __init__(self, app):
        self._app = app
        self.skipped = []

    def include_router(self, router, skip_pat=False, skip_brain=False, **kwargs):
        self._app.include_router(router, **kwargs)
        self.skipped.append((skip_pat, skip_brain, kwargs.get("prefix") or getattr(router, "prefix", "")))


def _source(store, **overrides):
    payload = {
        "name": "Help center",
        "source_type": "zendesk_guide",
        "sync_mode": "webhook",
        "webhook_secret_env": "ZENDESK_GUIDE_WEBHOOK_SECRET",
        "skip_enrichment": True,
    }
    payload.update(overrides)
    return store.create("brain", payload)


class ZendeskConnectorTests(unittest.TestCase):
    def setUp(self):
        self.connector = ZendeskGuideConnector()

    def test_parses_article_published_webhook(self):
        body = json.dumps(
            {
                "type": "zen:event-type:article.published",
                "detail": {
                    "id": 4410,
                    "title": "Refund window",
                    "body": "Refunds are accepted within 30 days.",
                    "html_url": "https://example.zendesk.com/hc/articles/4410",
                    "updated_at": "2026-11-25T12:00:00Z",
                },
            }
        ).encode("utf-8")
        events = self.connector.parse_webhook({}, body)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].action, "upsert")
        self.assertEqual(events[0].document_id, "4410")
        self.assertEqual(events[0].revision, "2026-11-25T12:00:00Z")
        self.assertIn("30 days", events[0].body)

    def test_deleted_article_is_a_delete_event(self):
        body = json.dumps(
            {
                "type": "zen:event-type:article.deleted",
                "detail": {"id": "4410", "updated_at": "2026-11-25T13:00:00Z"},
            }
        ).encode("utf-8")
        events = self.connector.parse_webhook({}, body)
        self.assertEqual(events[0].action, "delete")

    def test_draft_article_is_treated_as_delete(self):
        body = json.dumps(
            {
                "type": "zen:event-type:article.updated",
                "detail": {"id": "9", "title": "Hidden", "body": "n/a", "draft": True},
            }
        ).encode("utf-8")
        events = self.connector.parse_webhook({}, body)
        self.assertEqual(events[0].action, "delete")

    def test_invalid_json_is_rejected(self):
        with self.assertRaises(WebhookRejected):
            self.connector.parse_webhook({}, b"not-json")

    def test_verifies_generic_and_zendesk_signatures(self):
        secret = "hook-secret"
        body = b'{"detail":{"id":1}}'
        generic = generic_hmac_hex(secret, body)
        self.assertTrue(
            self.connector.verify_webhook(
                {"X-Source-Sync-Signature": generic}, body, secret
            )
        )
        self.assertFalse(
            self.connector.verify_webhook(
                {"X-Source-Sync-Signature": "deadbeef"}, body, secret
            )
        )
        timestamp = "2026-11-25T12:00:00Z"
        signature = zendesk_signature(secret, timestamp, body)
        self.assertTrue(
            self.connector.verify_webhook(
                {
                    "X-Zendesk-Webhook-Signature": signature,
                    "X-Zendesk-Webhook-Signature-Timestamp": timestamp,
                },
                body,
                secret,
            )
        )

    def test_poll_lists_articles_through_injected_http(self):
        pages = {
            "https://example.zendesk.com/api/v2/help_center/articles.json": {
                "articles": [
                    {
                        "id": 1,
                        "title": "A",
                        "body": "first",
                        "updated_at": "2026-11-01T00:00:00Z",
                    }
                ],
                "next_page": None,
            }
        }

        def getter(url, headers):
            self.assertIn("Authorization", headers)
            return pages[url]

        connector = ZendeskGuideConnector(http_get=getter)
        missing = {
            "base_url": "https://example.zendesk.com",
            "api_token_env": "ZENDESK_GUIDE_API_TOKEN",
        }
        with self.assertRaises(SourceFetchError):
            connector.list_documents(missing)
        self.assertIsNone(os.getenv("ZENDESK_GUIDE_API_TOKEN"))
        with patch.dict(os.environ, {"ZENDESK_GUIDE_API_TOKEN": "token"}):
            events = connector.list_documents(missing)
        self.assertEqual([item.document_id for item in events], ["1"])


class SourceSyncStoreTests(unittest.TestCase):
    def test_create_list_and_last_synced(self):
        store = SourceSyncStore()
        record = _source(store, name="Refunds")
        self.assertIsNone(record["last_synced"])
        public = public_source(record)
        self.assertNotIn("api_email", public)
        updated = store.mark_synced("brain", record["id"], "2026-11-25T12:00:00Z")
        self.assertEqual(updated["last_synced"], "2026-11-25T12:00:00Z")
        listed = store.list("brain")
        self.assertEqual(listed[0]["last_synced"], "2026-11-25T12:00:00Z")

    def test_revision_claim_is_idempotent(self):
        store = SourceSyncStore()
        self.assertTrue(store.claim_revision("src:doc:1:upsert", "brain"))
        self.assertFalse(store.claim_revision("src:doc:1:upsert", "brain"))
        store.release_revision("src:doc:1:upsert", "brain")
        self.assertTrue(store.claim_revision("src:doc:1:upsert", "brain"))

    def test_poll_due_uses_interval(self):
        store = SourceSyncStore()
        record = _source(store, sync_mode="poll", poll_interval_seconds=3600)
        self.assertTrue(store.due_for_poll(record, now=1_000_000))
        store.mark_synced("brain", record["id"], "1970-01-12T13:46:40Z")
        record = store.get("brain", record["id"])
        self.assertFalse(store.due_for_poll(record, now=1_000_000 + 10))
        self.assertTrue(store.due_for_poll(record, now=1_000_000 + 3601))


class SourceSyncApplyTests(unittest.TestCase):
    def test_upsert_queues_ingest_with_source_key_and_sets_last_synced(self):
        store = SourceSyncStore()
        source = _source(store)
        queued = []

        def queue_ingest(payload):
            queued.append(payload)
            return "task-1"

        result = apply_event(
            event_from_payload(
                {
                    "action": "upsert",
                    "document_id": "4410",
                    "revision": "rev-1",
                    "title": "Refunds",
                    "body": "Refunds are accepted within 30 days.",
                    "occurred_at": "2026-11-25T12:00:00Z",
                }
            ),
            source,
            store=store,
            queue_ingest=queue_ingest,
        )
        self.assertEqual(result["status"], "queued")
        self.assertEqual(result["source_key"], "zendesk_guide:4410")
        self.assertEqual(queued[0]["meta_keys"]["source_key"], "zendesk_guide:4410")
        self.assertEqual(queued[0]["source_timestamp"], "2026-11-25T12:00:00Z")
        parsed = IngestionTaskArgs(**queued[0])
        self.assertEqual(parsed.meta_keys["source_key"], "zendesk_guide:4410")
        self.assertTrue(parsed.skip_enrichment)
        self.assertEqual(
            store.get("brain", source["id"])["last_synced"], "2026-11-25T12:00:00Z"
        )
        again = apply_event(
            event_from_payload(
                {
                    "action": "upsert",
                    "document_id": "4410",
                    "revision": "rev-1",
                    "title": "Refunds",
                    "body": "Refunds are accepted within 30 days.",
                }
            ),
            source,
            store=store,
            queue_ingest=queue_ingest,
        )
        self.assertEqual(again["status"], "duplicate")
        self.assertEqual(len(queued), 1)

    def test_delete_closes_passages_and_facts_at_deletion_time(self):
        store = SourceSyncStore()
        source = _source(store)
        live = TextChunk(
            id="chunk-1",
            text="Refunds are accepted within 30 days.",
            metadata={"source_key": "zendesk_guide:4410"},
        )
        data = FakeData([live])
        predicate = SimpleNamespace(
            uuid="fact-1",
            properties={"source_chunk_ids": ["chunk-1"], "uuid": "fact-1"},
        )
        graph = FakeGraph([(SimpleNamespace(), predicate, SimpleNamespace())])
        result = apply_event(
            event_from_payload(
                {
                    "action": "delete",
                    "document_id": "4410",
                    "revision": "rev-gone",
                    "occurred_at": "2026-11-25T15:00:00Z",
                }
            ),
            source,
            store=store,
            data=data,
            graph=graph,
        )
        self.assertEqual(result["status"], "deleted")
        self.assertEqual(result["valid_to"], "2026-11-25T15:00:00Z")
        self.assertEqual(live.metadata["valid_to"], "2026-11-25T15:00:00Z")
        self.assertEqual(graph.updated[0]["new_properties"]["valid_to"], "2026-11-25T15:00:00Z")
        self.assertTrue(graph.updated[0]["expected_open"])
        self.assertEqual(graph.updated[0]["uuid"], "fact-1")

    def test_failed_apply_releases_revision_for_retry(self):
        store = SourceSyncStore()
        source = _source(store)

        def boom(_payload):
            raise RuntimeError("queue down")

        with self.assertRaises(RuntimeError):
            apply_event(
                event_from_payload(
                    {
                        "action": "upsert",
                        "document_id": "1",
                        "revision": "r1",
                        "body": "hello",
                    }
                ),
                source,
                store=store,
                queue_ingest=boom,
            )
        retried = apply_event(
            event_from_payload(
                {
                    "action": "upsert",
                    "document_id": "1",
                    "revision": "r1",
                    "body": "hello",
                }
            ),
            source,
            store=store,
            queue_ingest=lambda payload: "task-2",
        )
        self.assertEqual(retried["status"], "queued")

    def test_dead_letter_is_listed_without_secrets(self):
        store = SourceSyncStore()
        source = _source(store)
        item = dead_letter(
            source,
            {"document_id": "1", "revision": "r", "action": "upsert"},
            "queue down",
            store=store,
        )
        self.assertEqual(item["error"], "queue down")
        listed = store.list_dead_letters("brain")
        self.assertEqual(listed[0]["document_id"], "1")
        blob = json.dumps(listed)
        self.assertNotIn("hook-secret", blob)
        self.assertNotIn("api_email", blob)
        self.assertNotIn("api_token_env", blob)


class CloseCitingFactsTests(unittest.TestCase):
    def test_skips_edges_that_do_not_cite_the_closed_chunks(self):
        other = SimpleNamespace(
            uuid="fact-other",
            properties={"source_chunk_ids": ["chunk-b"], "uuid": "fact-other"},
        )
        graph = FakeGraph([(SimpleNamespace(), other, SimpleNamespace())])
        closed = close_citing_facts(graph, "brain", ["chunk-a"], "2026-11-25T15:00:00Z")
        self.assertEqual(closed, [])
        self.assertEqual(graph.updated, [])


class SourceSyncRouteTests(unittest.TestCase):
    def setUp(self):
        self.store = SourceSyncStore()
        self.app = FastAPI()

        @self.app.middleware("http")
        async def set_brain(request: Request, call_next):
            request.state.brain_id = "brain"
            return await call_next(request)

        self.queued = []
        self.patches = [
            patch("source_sync_api.default_store", lambda: self.store),
            patch("source_sync_store.default_store", lambda: self.store),
            patch(
                "source_sync_api._enqueue_apply",
                lambda payload: self.queued.append(payload) or "celery-1",
            ),
            patch("source_sync_api._enqueue_poll", lambda source_id: f"poll-{source_id}"),
        ]
        for item in self.patches:
            item.start()
        self.app.include_router(create_admin_router())
        self.app.include_router(create_webhook_router())
        self.client = TestClient(self.app)

    def tearDown(self):
        for item in self.patches:
            item.stop()

    def test_create_and_list_exposes_last_synced(self):
        created = self.client.post(
            "/source-sync/sources",
            json={"name": "Help center", "source_type": "zendesk_guide"},
        )
        self.assertEqual(created.status_code, 201)
        body = created.json()
        self.assertIsNone(body["last_synced"])
        self.assertTrue(body["webhook_path"].startswith("/source-sync/webhooks/"))
        listed = self.client.get("/source-sync/sources")
        self.assertEqual(listed.status_code, 200)
        self.assertEqual(listed.json()["total"], 1)
        self.assertIsNone(listed.json()["sources"][0]["last_synced"])

    def test_webhook_requires_signature_and_enqueues_events(self):
        os.environ["ZENDESK_GUIDE_WEBHOOK_SECRET"] = "hook-secret"
        self.addCleanup(lambda: os.environ.pop("ZENDESK_GUIDE_WEBHOOK_SECRET", None))
        created = self.client.post(
            "/source-sync/sources",
            json={
                "name": "Help center",
                "source_type": "zendesk_guide",
                "webhook_secret_env": "ZENDESK_GUIDE_WEBHOOK_SECRET",
            },
        )
        source_id = created.json()["id"]
        payload = {
            "type": "zen:event-type:article.published",
            "detail": {
                "id": "77",
                "title": "Hours",
                "body": "Open 9 to 5",
                "updated_at": "2026-11-25T12:00:00Z",
            },
        }
        raw = json.dumps(payload).encode("utf-8")
        denied = self.client.post(f"/source-sync/webhooks/{source_id}", content=raw)
        self.assertEqual(denied.status_code, 401)
        signature = generic_hmac_hex("hook-secret", raw)
        accepted = self.client.post(
            f"/source-sync/webhooks/{source_id}",
            content=raw,
            headers={"X-Source-Sync-Signature": signature},
        )
        self.assertEqual(accepted.status_code, 200)
        self.assertEqual(accepted.json()["count"], 1)
        self.assertEqual(self.queued[0]["document_id"], "77")
        self.assertEqual(self.queued[0]["action"], "upsert")


class PluginRegisterTests(unittest.TestCase):
    def test_register_mounts_webhook_without_pat(self):
        app = FastAPI()
        context = FakeContext(app)
        register(context)
        prefixes = [item[2] for item in context.skipped]
        self.assertIn("/source-sync", prefixes)
        self.assertIn("/source-sync/webhooks", prefixes)
        webhook = [item for item in context.skipped if item[2] == "/source-sync/webhooks"][0]
        self.assertTrue(webhook[0])
        self.assertTrue(webhook[1])


class CeleryDiscoveryTests(unittest.TestCase):
    def test_plugin_declares_queues_routes_and_beat(self):
        path = PLUGIN_DIR / "workers" / "celery.py"
        spec = importlib.util.spec_from_file_location(
            "source_sync_celery_fixture", path
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        names = {getattr(queue, "name", None) for queue in module.QUEUES}
        self.assertIn("source_sync", names)
        self.assertIn("source_sync_dead", names)
        self.assertEqual(module.ROUTES["source_sync.apply_event"]["queue"], "source_sync")
        self.assertIn("source-sync-recheck", module.BEAT_SCHEDULE)
        self.assertEqual(
            module.BEAT_SCHEDULE["source-sync-recheck"]["task"],
            "source_sync.recheck_due_sources",
        )

    def test_manifest_is_discovered(self):
        catalog = PluginManifestCatalog(PLUGIN_DIR.parent)
        names = {item.name for item in catalog.discover(validate=True)}
        self.assertIn("source_sync", names)

    def test_beat_from_temp_plugin(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plugin = root / "demo"
            workers = plugin / "workers"
            workers.mkdir(parents=True)
            (workers / "celery.py").write_text(
                "from kombu import Queue\n"
                "QUEUES = (Queue('demo', routing_key='demo'),)\n"
                "ROUTES = {'demo.ping': {'queue': 'demo'}}\n"
                "BEAT_SCHEDULE = {'demo-tick': {'task': 'demo.ping', 'schedule': 30.0}}\n",
                encoding="utf-8",
            )
            (workers / "tasks.py").write_text("PING = True\n", encoding="utf-8")
            queues, routes, beat = discover_plugin_celery(root)
            self.assertEqual(getattr(queues[0], "name", None), "demo")
            self.assertEqual(routes["demo.ping"]["queue"], "demo")
            self.assertEqual(beat["demo-tick"]["task"], "demo.ping")
