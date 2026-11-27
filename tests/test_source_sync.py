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

from source_sync_apply import (
    DocumentBusy,
    apply_event,
    apply_poll,
    approve_pending_deletes,
    close_citing_facts,
    commit_upsert_head,
    dead_letter,
    event_from_payload,
    poll_delete_abort_reason,
    record_failed_event,
)
from source_sync_connector import (
    SourceFetchError,
    WebhookRejected,
    ZendeskGuideConnector,
    generic_hmac_hex,
    list_max_pages,
    zendesk_signature,
)
from source_sync_store import (
    AdapterDurable,
    SourceSyncStore,
    pending_delete_digest,
    public_source,
)
from src.core.saving.validity import format_instant, utc_now
from source_sync_api import (
    create_admin_router,
    create_webhook_router,
    lookup_source_for_brain,
    source_sync_approve_deletes,
    source_sync_get_source,
)
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

    def get_text_chunks_by_ids(self, ids, with_observations, brain_id):
        wanted = {str(item) for item in ids}
        return [chunk for chunk in self.chunks if str(chunk.id) in wanted], []


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


def _fresh_generic_headers(secret: str, body: bytes) -> dict[str, str]:
    timestamp = format_instant(utc_now())
    return {
        "X-Source-Sync-Signature": generic_hmac_hex(secret, body, timestamp),
        "X-Source-Sync-Timestamp": timestamp,
    }


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
        headers = _fresh_generic_headers(secret, body)
        self.assertTrue(self.connector.verify_webhook(headers, body, secret))
        bad = dict(headers)
        bad["X-Source-Sync-Signature"] = "deadbeef"
        self.assertFalse(self.connector.verify_webhook(bad, body, secret))
        timestamp = format_instant(utc_now())
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

    def test_rejects_stale_webhook_timestamp(self):
        secret = "hook-secret"
        body = b'{"detail":{"id":1}}'
        timestamp = "2020-01-01T00:00:00Z"
        headers = {
            "X-Source-Sync-Signature": generic_hmac_hex(secret, body, timestamp),
            "X-Source-Sync-Timestamp": timestamp,
        }
        self.assertFalse(self.connector.verify_webhook(headers, body, secret))
        self.assertFalse(
            self.connector.verify_webhook(
                {"X-Source-Sync-Signature": generic_hmac_hex(secret, body)},
                body,
                secret,
            )
        )

    def test_poll_lists_articles_through_injected_http(self):
        page = {
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

        def getter(url, headers):
            self.assertIn("Authorization", headers)
            self.assertIn("page", url)
            self.assertIn("size", url)
            return page

        connector = ZendeskGuideConnector(http_get=getter)
        missing = {
            "base_url": "https://example.zendesk.com",
            "api_token_env": "ZENDESK_GUIDE_API_TOKEN",
        }
        with self.assertRaises(SourceFetchError):
            connector.list_documents(missing)
        self.assertIsNone(os.getenv("ZENDESK_GUIDE_API_TOKEN"))
        with patch.dict(os.environ, {"ZENDESK_GUIDE_API_TOKEN": "token"}):
            listing = connector.list_documents(missing)
        self.assertEqual([item.document_id for item in listing.events], ["1"])
        self.assertFalse(listing.truncated)

    def test_list_documents_raises_when_articles_list_is_missing(self):
        def getter(url, headers):
            return {"next_page": None}

        connector = ZendeskGuideConnector(http_get=getter)
        with patch.dict(os.environ, {"ZENDESK_GUIDE_API_TOKEN": "token"}):
            with self.assertRaises(SourceFetchError) as raised:
                connector.list_documents(
                    {
                        "base_url": "https://example.zendesk.com",
                        "api_token_env": "ZENDESK_GUIDE_API_TOKEN",
                    }
                )
        self.assertIn("articles list", str(raised.exception))

    def test_list_documents_truncates_when_page_limit_is_hit(self):
        def getter(url, headers):
            return {
                "articles": [{"id": 1, "title": "A", "body": "keep"}],
                "links": {"next": url},
                "meta": {"has_more": True},
            }

        connector = ZendeskGuideConnector(http_get=getter)
        with patch.dict(
            os.environ,
            {
                "ZENDESK_GUIDE_API_TOKEN": "token",
                "SOURCE_SYNC_LIST_MAX_PAGES": "2",
            },
        ):
            self.assertEqual(list_max_pages(), 2)
            listing = connector.list_documents(
                {
                    "base_url": "https://example.zendesk.com",
                    "api_token_env": "ZENDESK_GUIDE_API_TOKEN",
                }
            )
        self.assertTrue(listing.truncated)
        self.assertEqual(listing.pages_fetched, 2)
        self.assertEqual(len(listing.events), 2)


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

    def test_document_lock_serializes_the_same_document(self):
        store = SourceSyncStore()
        self.assertTrue(store.acquire_document_lock("brain", "src", "doc-1"))
        self.assertFalse(store.acquire_document_lock("brain", "src", "doc-1"))
        store.release_document_lock("brain", "src", "doc-1")
        self.assertTrue(store.acquire_document_lock("brain", "src", "doc-1"))

    def test_webhook_without_credentials_is_not_due_for_poll(self):
        store = SourceSyncStore()
        webhook = _source(store, sync_mode="webhook")
        self.assertFalse(store.due_for_poll(webhook, now=1_000_000))
        catchup = _source(
            store,
            name="Catchup",
            sync_mode="webhook",
            api_token_env="ZENDESK_GUIDE_API_TOKEN",
            base_url="https://example.zendesk.com",
        )
        self.assertTrue(store.due_for_poll(catchup, now=1_000_000))

    def test_durable_adapter_keeps_sources_without_cache(self):
        class FakeAdapter:
            def __init__(self):
                self._data = {}

            def source_sync_put(self, brain_id, bucket, item_id, document):
                self._data[(brain_id, bucket, item_id)] = dict(document)

            def source_sync_get(self, brain_id, bucket, item_id):
                value = self._data.get((brain_id, bucket, item_id))
                return dict(value) if value else None

            def source_sync_list(self, brain_id, bucket):
                found = {}
                for (stored_brain, stored_bucket, item_id), document in self._data.items():
                    if stored_brain == brain_id and stored_bucket == bucket:
                        found[item_id] = dict(document)
                return found

            def source_sync_delete(self, brain_id, bucket, item_id):
                return self._data.pop((brain_id, bucket, item_id), None) is not None

        durable = AdapterDurable(FakeAdapter())
        store = SourceSyncStore(cache=None, durable=durable)
        record = _source(store, name="Durable")
        revived = SourceSyncStore(cache=None, durable=durable)
        fetched = revived.get("brain", record["id"])
        self.assertEqual(fetched["name"], "Durable")
        self.assertIsNone(revived.cache)

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
        self.assertIsNone(store.document_head("brain", source["id"], "4410"))
        retry = apply_event(
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
        self.assertEqual(retry["status"], "queued")
        self.assertEqual(len(queued), 2)
        committed = commit_upsert_head(
            brain_id="brain",
            source_id=source["id"],
            document_id="4410",
            revision="rev-1",
            occurred_at="2026-11-25T12:00:00Z",
            source_key="zendesk_guide:4410",
            store=store,
        )
        self.assertEqual(committed["status"], "committed")
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
        self.assertEqual(len(queued), 2)

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

    def test_older_update_after_newer_head_is_stale(self):
        store = SourceSyncStore()
        source = _source(store)
        apply_event(
            event_from_payload(
                {
                    "action": "upsert",
                    "document_id": "4410",
                    "revision": "rev-2",
                    "body": "new",
                    "occurred_at": "2026-11-25T14:00:00Z",
                }
            ),
            source,
            store=store,
            queue_ingest=lambda payload: "task-new",
        )
        commit_upsert_head(
            brain_id="brain",
            source_id=source["id"],
            document_id="4410",
            revision="rev-2",
            occurred_at="2026-11-25T14:00:00Z",
            source_key="zendesk_guide:4410",
            store=store,
        )
        stale = apply_event(
            event_from_payload(
                {
                    "action": "upsert",
                    "document_id": "4410",
                    "revision": "rev-1",
                    "body": "old",
                    "occurred_at": "2026-11-25T12:00:00Z",
                }
            ),
            source,
            store=store,
            queue_ingest=lambda payload: "task-old",
        )
        self.assertEqual(stale["status"], "stale")

    def test_update_after_delete_is_stale_when_older(self):
        store = SourceSyncStore()
        source = _source(store)
        apply_event(
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
            data=FakeData(),
        )
        late = apply_event(
            event_from_payload(
                {
                    "action": "upsert",
                    "document_id": "4410",
                    "revision": "rev-1",
                    "body": "resurrect",
                    "occurred_at": "2026-11-25T12:00:00Z",
                }
            ),
            source,
            store=store,
            queue_ingest=lambda payload: "task-late",
        )
        self.assertEqual(late["status"], "stale")

    def test_busy_document_raises_without_applying(self):
        store = SourceSyncStore()
        source = _source(store)
        self.assertTrue(store.acquire_document_lock("brain", source["id"], "4410"))
        with self.assertRaises(DocumentBusy):
            apply_event(
                event_from_payload(
                    {
                        "action": "upsert",
                        "document_id": "4410",
                        "revision": "rev-1",
                        "body": "hello",
                    }
                ),
                source,
                store=store,
                queue_ingest=lambda payload: "task-busy",
            )

    def test_poll_emits_deletes_for_missing_documents(self):
        store = SourceSyncStore()
        source = _source(
            store,
            sync_mode="poll",
            base_url="https://example.zendesk.com",
            api_token_env="ZENDESK_GUIDE_API_TOKEN",
        )
        store.set_seen_ids("brain", source["id"], {"1", "2"})
        queued = []

        def getter(url, headers):
            return {
                "articles": [
                    {
                        "id": 1,
                        "title": "Keep",
                        "body": "still here",
                        "updated_at": "2026-11-25T12:00:00Z",
                    }
                ],
                "next_page": None,
            }

        with patch.dict(os.environ, {"ZENDESK_GUIDE_API_TOKEN": "token"}):
            from source_sync_connector import ZendeskGuideConnector, CONNECTORS

            original = CONNECTORS["zendesk_guide"]
            CONNECTORS["zendesk_guide"] = ZendeskGuideConnector(http_get=getter)
            try:
                result = apply_poll(
                    source,
                    store=store,
                    enqueue_event=lambda payload: queued.append(payload),
                )
            finally:
                CONNECTORS["zendesk_guide"] = original
        self.assertEqual(result["deleted"], 1)
        self.assertEqual(result["deleted_ids"], ["2"])
        actions = {item["document_id"]: item["action"] for item in queued}
        self.assertEqual(actions["2"], "delete")
        self.assertEqual(actions["1"], "upsert")
        self.assertEqual(store.get_seen_ids("brain", source["id"]), {"1"})
        self.assertIsNone(result["retirement_aborted"])

    def test_poll_does_not_emit_deletes_when_listing_is_incomplete(self):
        store = SourceSyncStore()
        source = _source(
            store,
            sync_mode="poll",
            base_url="https://example.zendesk.com",
            api_token_env="ZENDESK_GUIDE_API_TOKEN",
        )
        store.set_seen_ids("brain", source["id"], {"1", "2"})
        queued = []

        def getter(url, headers):
            return {"next_page": None}

        with patch.dict(os.environ, {"ZENDESK_GUIDE_API_TOKEN": "token"}):
            from source_sync_connector import ZendeskGuideConnector, CONNECTORS

            original = CONNECTORS["zendesk_guide"]
            CONNECTORS["zendesk_guide"] = ZendeskGuideConnector(http_get=getter)
            try:
                with self.assertRaises(SourceFetchError):
                    apply_poll(
                        source,
                        store=store,
                        enqueue_event=lambda payload: queued.append(payload),
                    )
            finally:
                CONNECTORS["zendesk_guide"] = original
        self.assertEqual(queued, [])
        self.assertEqual(store.get_seen_ids("brain", source["id"]), {"1", "2"})

    def test_poll_skips_deletes_when_cap_is_exceeded(self):
        store = SourceSyncStore()
        source = _source(
            store,
            sync_mode="poll",
            base_url="https://example.zendesk.com",
            api_token_env="ZENDESK_GUIDE_API_TOKEN",
        )
        store.set_seen_ids("brain", source["id"], {str(index) for index in range(1, 11)})
        queued = []

        def getter(url, headers):
            return {
                "articles": [
                    {
                        "id": 1,
                        "title": "Keep",
                        "body": "still here",
                        "updated_at": "2026-11-25T12:00:00Z",
                    }
                ],
                "next_page": None,
            }

        with patch.dict(
            os.environ,
            {
                "ZENDESK_GUIDE_API_TOKEN": "token",
                "SOURCE_SYNC_DELETE_MAX_RATIO": "0.2",
                "SOURCE_SYNC_DELETE_MAX_PAGES": "50",
            },
        ):
            from source_sync_connector import ZendeskGuideConnector, CONNECTORS

            original = CONNECTORS["zendesk_guide"]
            CONNECTORS["zendesk_guide"] = ZendeskGuideConnector(http_get=getter)
            try:
                result = apply_poll(
                    source,
                    store=store,
                    enqueue_event=lambda payload: queued.append(payload),
                )
            finally:
                CONNECTORS["zendesk_guide"] = original
        self.assertEqual(result["deleted"], 0)
        self.assertTrue(result["retirement_aborted"])
        self.assertIn("exceeds retirement cap", result["retirement_aborted"])
        self.assertEqual([item["action"] for item in queued], ["upsert"])
        self.assertEqual(
            store.get_seen_ids("brain", source["id"]),
            {str(index) for index in range(1, 11)},
        )
        self.assertEqual(poll_delete_abort_reason(10, 9) is not None, True)
        status = store.get("brain", source["id"])
        self.assertEqual(status["pending_delete_count"], 9)
        self.assertTrue(status["pending_delete_first_seen"])
        self.assertIn("exceeds retirement cap", status["pending_delete_reason"])
        letters = store.list_dead_letters("brain")
        self.assertEqual(letters[0]["kind"], "retirement_blocked")
        self.assertEqual(letters[0]["pending_delete_count"], 9)
        first_seen = status["pending_delete_first_seen"]
        queued.clear()
        with patch.dict(
            os.environ,
            {
                "ZENDESK_GUIDE_API_TOKEN": "token",
                "SOURCE_SYNC_DELETE_MAX_RATIO": "0.2",
                "SOURCE_SYNC_DELETE_MAX_PAGES": "50",
            },
        ):
            from source_sync_connector import ZendeskGuideConnector, CONNECTORS

            original = CONNECTORS["zendesk_guide"]
            CONNECTORS["zendesk_guide"] = ZendeskGuideConnector(http_get=getter)
            try:
                apply_poll(
                    source,
                    store=store,
                    enqueue_event=lambda payload: queued.append(payload),
                )
            finally:
                CONNECTORS["zendesk_guide"] = original
        self.assertEqual(store.get("brain", source["id"])["pending_delete_first_seen"], first_seen)
        self.assertEqual(len(store.list_dead_letters("brain")), 1)

    def test_poll_applies_upserts_when_listing_is_truncated(self):
        store = SourceSyncStore()
        source = _source(
            store,
            sync_mode="poll",
            base_url="https://example.zendesk.com",
            api_token_env="ZENDESK_GUIDE_API_TOKEN",
        )
        store.set_seen_ids("brain", source["id"], {"1", "2"})
        queued = []

        def getter(url, headers):
            return {
                "articles": [
                    {
                        "id": 1,
                        "title": "Keep",
                        "body": "still here",
                        "updated_at": "2026-11-25T12:00:00Z",
                    }
                ],
                "links": {"next": url},
                "meta": {"has_more": True},
            }

        with patch.dict(
            os.environ,
            {
                "ZENDESK_GUIDE_API_TOKEN": "token",
                "SOURCE_SYNC_LIST_MAX_PAGES": "1",
            },
        ):
            from source_sync_connector import ZendeskGuideConnector, CONNECTORS

            original = CONNECTORS["zendesk_guide"]
            CONNECTORS["zendesk_guide"] = ZendeskGuideConnector(http_get=getter)
            try:
                result = apply_poll(
                    source,
                    store=store,
                    enqueue_event=lambda payload: queued.append(payload),
                )
            finally:
                CONNECTORS["zendesk_guide"] = original
        self.assertTrue(result["listing_truncated"])
        self.assertEqual(result["deleted"], 0)
        self.assertEqual([item["action"] for item in queued], ["upsert"])
        self.assertEqual(store.get_seen_ids("brain", source["id"]), {"1", "2"})
        self.assertTrue(store.get("brain", source["id"])["listing_truncated"])
        self.assertEqual(store.get("brain", source["id"])["pending_delete_count"], 0)

    def test_approve_pending_deletes_allows_over_cap_poll(self):
        store = SourceSyncStore()
        source = _source(
            store,
            sync_mode="poll",
            base_url="https://example.zendesk.com",
            api_token_env="ZENDESK_GUIDE_API_TOKEN",
        )
        store.set_seen_ids("brain", source["id"], {str(index) for index in range(1, 11)})
        queued = []

        def getter(url, headers):
            return {
                "articles": [
                    {
                        "id": 1,
                        "title": "Keep",
                        "body": "still here",
                        "updated_at": "2026-11-25T12:00:00Z",
                    }
                ],
                "next_page": None,
            }

        def poll_once():
            from source_sync_connector import ZendeskGuideConnector, CONNECTORS

            original = CONNECTORS["zendesk_guide"]
            CONNECTORS["zendesk_guide"] = ZendeskGuideConnector(http_get=getter)
            try:
                return apply_poll(
                    source,
                    store=store,
                    enqueue_event=lambda payload: queued.append(payload),
                )
            finally:
                CONNECTORS["zendesk_guide"] = original

        with patch.dict(
            os.environ,
            {
                "ZENDESK_GUIDE_API_TOKEN": "token",
                "SOURCE_SYNC_DELETE_MAX_RATIO": "0.2",
                "SOURCE_SYNC_DELETE_MAX_PAGES": "50",
            },
        ):
            blocked = poll_once()
            self.assertTrue(blocked["retirement_aborted"])
            status = store.get("brain", source["id"])
            approve_pending_deletes(
                "brain",
                source["id"],
                count=int(status["pending_delete_count"]),
                digest=status["pending_delete_hash"],
                store=store,
            )
            queued.clear()
            allowed = poll_once()
        self.assertIsNone(allowed["retirement_aborted"])
        self.assertEqual(allowed["deleted"], 9)
        self.assertEqual(store.get("brain", source["id"])["pending_delete_count"], 0)
        self.assertFalse(store.get("brain", source["id"])["approve_pending_deletes"])
        self.assertEqual(store.get_seen_ids("brain", source["id"]), {"1"})

    def test_changed_pending_set_resets_approval(self):
        store = SourceSyncStore()
        source = _source(
            store,
            sync_mode="poll",
            base_url="https://example.zendesk.com",
            api_token_env="ZENDESK_GUIDE_API_TOKEN",
        )
        store.set_seen_ids("brain", source["id"], {str(index) for index in range(1, 11)})
        queued = []
        live_ids = {"1"}

        def getter(url, headers):
            return {
                "articles": [
                    {
                        "id": int(item),
                        "title": "Keep",
                        "body": "still here",
                        "updated_at": "2026-11-25T12:00:00Z",
                    }
                    for item in sorted(live_ids)
                ],
                "next_page": None,
            }

        def poll_once():
            from source_sync_connector import ZendeskGuideConnector, CONNECTORS

            original = CONNECTORS["zendesk_guide"]
            CONNECTORS["zendesk_guide"] = ZendeskGuideConnector(http_get=getter)
            try:
                return apply_poll(
                    source,
                    store=store,
                    enqueue_event=lambda payload: queued.append(payload),
                )
            finally:
                CONNECTORS["zendesk_guide"] = original

        with patch.dict(
            os.environ,
            {
                "ZENDESK_GUIDE_API_TOKEN": "token",
                "SOURCE_SYNC_DELETE_MAX_RATIO": "0.2",
                "SOURCE_SYNC_DELETE_MAX_PAGES": "50",
            },
        ):
            blocked = poll_once()
            self.assertTrue(blocked["retirement_aborted"])
            status = store.get("brain", source["id"])
            approve_pending_deletes(
                "brain",
                source["id"],
                count=int(status["pending_delete_count"]),
                digest=status["pending_delete_hash"],
                store=store,
            )
            live_ids = {"1", "2"}
            queued.clear()
            skipped = poll_once()
        self.assertTrue(skipped["retirement_aborted"])
        self.assertEqual(skipped["deleted"], 0)
        self.assertFalse(store.get("brain", source["id"])["approve_pending_deletes"])
        self.assertEqual(store.get("brain", source["id"])["pending_delete_count"], 8)

    def test_older_ingest_commit_does_not_overwrite_newer_head(self):
        store = SourceSyncStore()
        source = _source(store)
        apply_event(
            event_from_payload(
                {
                    "action": "upsert",
                    "document_id": "4410",
                    "revision": "rev-old",
                    "body": "old",
                    "occurred_at": "2026-11-25T12:00:00Z",
                }
            ),
            source,
            store=store,
            queue_ingest=lambda payload: "task-old",
        )
        apply_event(
            event_from_payload(
                {
                    "action": "upsert",
                    "document_id": "4410",
                    "revision": "rev-new",
                    "body": "new",
                    "occurred_at": "2026-11-25T14:00:00Z",
                }
            ),
            source,
            store=store,
            queue_ingest=lambda payload: "task-new",
        )
        commit_upsert_head(
            brain_id="brain",
            source_id=source["id"],
            document_id="4410",
            revision="rev-new",
            occurred_at="2026-11-25T14:00:00Z",
            source_key="zendesk_guide:4410",
            store=store,
        )
        skipped = commit_upsert_head(
            brain_id="brain",
            source_id=source["id"],
            document_id="4410",
            revision="rev-old",
            occurred_at="2026-11-25T12:00:00Z",
            source_key="zendesk_guide:4410",
            store=store,
        )
        self.assertEqual(skipped["status"], "stale")
        head = store.document_head("brain", source["id"], "4410")
        self.assertEqual(head["revision"], "rev-new")
        self.assertEqual(head["ingest_status"], "committed")

    def test_lock_race_is_dead_lettered_after_retries(self):
        store = SourceSyncStore()
        source = _source(store)
        payload = {
            "source_id": source["id"],
            "brain_id": "brain",
            "document_id": "4410",
            "revision": "rev-1",
            "action": "upsert",
        }
        busy = DocumentBusy("document 4410 is busy")
        skipped = record_failed_event(
            payload, busy, retries=4, max_retries=5, store=store
        )
        self.assertIsNone(skipped)
        recorded = record_failed_event(
            payload, busy, retries=5, max_retries=5, store=store
        )
        self.assertIsNotNone(recorded)
        self.assertIn("busy", recorded["error"])
        listed = store.list_dead_letters("brain")
        self.assertEqual(listed[0]["document_id"], "4410")
        self.assertEqual(listed[0]["source_id"], source["id"])

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
        self.assertIsNone(store.document_head("brain", source["id"], "1"))

    def test_ingest_payload_commits_head_after_source_save(self):
        from source_sync_apply import commit_upsert_head_from_payload

        store = SourceSyncStore()
        source = _source(store)
        apply_event(
            event_from_payload(
                {
                    "action": "upsert",
                    "document_id": "4410",
                    "revision": "rev-1",
                    "body": "hello",
                    "occurred_at": "2026-11-25T12:00:00Z",
                }
            ),
            source,
            store=store,
            queue_ingest=lambda payload: "task-1",
        )
        self.assertIsNone(store.document_head("brain", source["id"], "4410"))
        payload = SimpleNamespace(
            brain_id="brain",
            source_timestamp="2026-11-25T12:00:00Z",
            meta_keys={
                "source_sync_id": source["id"],
                "source_document_id": "4410",
                "source_revision": "rev-1",
                "source_key": "zendesk_guide:4410",
            },
        )
        result = commit_upsert_head_from_payload(payload, store=store)
        self.assertEqual(result["status"], "committed")
        self.assertEqual(
            store.document_head("brain", source["id"], "4410")["revision"], "rev-1"
        )

    def test_ingest_commit_hooks_run_registered_handlers(self):
        from src.core.saving.ingest_commit_hooks import (
            register_ingest_commit_hook,
            run_ingest_commit_hooks,
            unregister_ingest_commit_hook,
        )

        seen = []
        register_ingest_commit_hook(seen.append)
        try:
            run_ingest_commit_hooks("payload")
            self.assertEqual(seen, ["payload"])
        finally:
            unregister_ingest_commit_hook(seen.append)

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

    def test_keeps_facts_that_still_cite_a_live_source(self):
        live = TextChunk(id="chunk-b", text="other", metadata={"source_key": "zendesk_guide:other"})
        closed_chunk = TextChunk(
            id="chunk-a",
            text="gone",
            metadata={"source_key": "zendesk_guide:4410", "valid_to": "2026-11-25T15:00:00Z"},
        )
        data = FakeData([live, closed_chunk])
        both = SimpleNamespace(
            uuid="fact-both",
            properties={"source_chunk_ids": ["chunk-a", "chunk-b"], "uuid": "fact-both"},
        )
        graph = FakeGraph([(SimpleNamespace(), both, SimpleNamespace())])
        closed = close_citing_facts(
            graph, "brain", ["chunk-a"], "2026-11-25T15:00:00Z", data=data
        )
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
        headers = _fresh_generic_headers("hook-secret", raw)
        accepted = self.client.post(
            f"/source-sync/webhooks/{source_id}",
            content=raw,
            headers=headers,
        )
        self.assertEqual(accepted.status_code, 200)
        self.assertEqual(accepted.json()["count"], 1)
        self.assertEqual(self.queued[0]["document_id"], "77")
        self.assertEqual(self.queued[0]["action"], "upsert")

    def test_approve_deletes_requires_pending_and_queues_poll(self):
        created = self.client.post(
            "/source-sync/sources",
            json={"name": "Help center", "source_type": "zendesk_guide"},
        )
        source_id = created.json()["id"]
        empty = self.client.post(
            f"/source-sync/sources/{source_id}/approve-deletes",
            json={
                "pending_delete_count": 4,
                "pending_delete_hash": pending_delete_digest(["1", "2", "3", "4"]),
            },
        )
        self.assertEqual(empty.status_code, 409)
        missing = ["1", "2", "3", "4"]
        self.store.record_blocked_deletes(
            "brain", source_id, 4, "cap exceeded", missing_ids=missing
        )
        digest = pending_delete_digest(missing)
        mismatch = self.client.post(
            f"/source-sync/sources/{source_id}/approve-deletes",
            json={
                "pending_delete_count": 4,
                "pending_delete_hash": pending_delete_digest(["other"]),
            },
        )
        self.assertEqual(mismatch.status_code, 409)
        self.assertEqual(mismatch.json()["detail"], "pending deletions changed")
        approved = self.client.post(
            f"/source-sync/sources/{source_id}/approve-deletes",
            json={
                "pending_delete_count": 4,
                "pending_delete_hash": digest,
            },
        )
        self.assertEqual(approved.status_code, 200)
        self.assertTrue(approved.json()["approved"])
        self.assertEqual(approved.json()["task_id"], f"poll-{source_id}")
        self.assertEqual(approved.json()["pending_delete_hash"], digest)
        fetched = self.client.get(f"/source-sync/sources/{source_id}")
        self.assertEqual(fetched.json()["pending_delete_count"], 4)
        self.assertEqual(fetched.json()["pending_delete_hash"], digest)
        self.assertEqual(fetched.json()["pending_delete_ids"], missing)
        self.assertTrue(fetched.json()["approve_pending_deletes"])
        self.assertTrue(fetched.json()["pending_delete_first_seen"])

    def test_approve_deletes_is_not_found_for_another_brain(self):
        other = self.store.create(
            "other-brain",
            {
                "name": "Other",
                "source_type": "zendesk_guide",
                "sync_mode": "webhook",
            },
        )
        missing = ["1", "2"]
        self.store.record_blocked_deletes(
            "other-brain", other["id"], 2, "cap exceeded", missing_ids=missing
        )
        missing_get = self.client.get(f"/source-sync/sources/{other['id']}")
        self.assertEqual(missing_get.status_code, 404)
        denied = self.client.post(
            f"/source-sync/sources/{other['id']}/approve-deletes",
            json={
                "pending_delete_count": 2,
                "pending_delete_hash": pending_delete_digest(missing),
            },
        )
        self.assertEqual(denied.status_code, 404)


class SourceSyncBrainScopeTests(unittest.TestCase):
    def test_mcp_tools_cannot_read_or_approve_another_brain(self):
        store = SourceSyncStore()
        source_a = store.create(
            "brain-a",
            {
                "name": "Alpha",
                "source_type": "zendesk_guide",
                "sync_mode": "webhook",
            },
        )
        source_b = store.create(
            "brain-b",
            {
                "name": "Beta",
                "source_type": "zendesk_guide",
                "sync_mode": "webhook",
            },
        )
        self.assertEqual(
            lookup_source_for_brain(store, "brain-a", source_a["id"])["id"],
            source_a["id"],
        )
        with self.assertRaises(ValueError):
            lookup_source_for_brain(store, "brain-a", source_b["id"])
        store.record_blocked_deletes(
            "brain-b", source_b["id"], 1, "cap exceeded", missing_ids=["9"]
        )
        digest = pending_delete_digest(["9"])
        with (
            patch("source_sync_api._mcp_brain", lambda brain_id: "brain-a"),
            patch("source_sync_api._store", lambda: store),
            patch("source_sync_api._enqueue_poll", lambda source_id: f"poll-{source_id}"),
        ):
            fetched = source_sync_get_source(source_b["id"], brain_id="brain-a")
            self.assertEqual(fetched, {"ok": False, "error": "source not found"})
            approved = source_sync_approve_deletes(
                source_b["id"],
                1,
                digest,
                brain_id="brain-a",
            )
            self.assertEqual(approved["ok"], False)
            self.assertEqual(approved["error"], "source not found")
            own = source_sync_get_source(source_a["id"], brain_id="brain-a")
            self.assertTrue(own["ok"])
            self.assertEqual(own["source"]["id"], source_a["id"])


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
