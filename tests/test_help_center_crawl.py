import os
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from src.adapters.interfaces.crawler import CrawledPage
from src.constants.data import TextChunk
from src.constants.kg import Node, Predicate
from src.core.saving.help_center_crawl import (
    CrawlSpec,
    claim_crawl_spec,
    close_removed_pages,
    crawl_root_key,
    mark_finished,
    normalize_source_key,
    page_source_key,
    spec_is_due,
)
from src.services.api.constants.requests import CrawlSpecRequestBody


ENV_DEFAULTS = {
    "BRAINPAT_TOKEN": "test-token",
    "MODELS_MODE": "local",
    "EMBEDDINGS_LOCAL_MODEL": "local-model",
    "EMBEDDINGS_SMALL_MODEL": "small-model",
    "EMBEDDING_NODES_DIMENSION": "3",
    "EMBEDDING_TRIPLETS_DIMENSION": "3",
    "EMBEDDING_OBSERVATIONS_DIMENSION": "3",
    "EMBEDDING_DATA_DIMENSION": "3",
    "EMBEDDING_RELATIONSHIPS_DIMENSION": "3",
    "REDIS_HOST": "localhost",
    "REDIS_PORT": "6379",
    "NEO4J_HOST": "localhost",
    "NEO4J_PORT": "7687",
    "NEO4J_USERNAME": "neo4j",
    "NEO4J_PASSWORD": "password",
    "MILVUS_HOST": "localhost",
    "MILVUS_PORT": "19530",
    "MONGO_CONNECTION_STRING": "mongodb://localhost:27017",
    "CELERY_WORKER_CONCURRENCY": "1",
    "OLLAMA_HOST": "localhost",
    "OLLAMA_PORT": "11434",
    "OLLAMA_LLM_SMALL_MODEL": "small",
    "OLLAMA_LLM_LARGE_MODEL": "large",
    "PIPELINE_MODE": "accurate",
}
for key, value in ENV_DEFAULTS.items():
    os.environ.setdefault(key, value)


ROOT = "https://help.example.com/docs"
LIVE = "https://help.example.com/docs/refunds"
GONE = "https://help.example.com/docs/old-policy"
NOW = datetime(2026, 11, 25, 12, 0, tzinfo=timezone.utc)


def _spec(**kwargs):
    payload = {
        "brain_id": "brain-a",
        "url": ROOT,
        "max_pages": 20,
        "allowed_path_prefixes": ["/docs"],
        "interval_days": 7,
    }
    payload.update(kwargs)
    return CrawlSpec.model_validate(payload)


def _chunk(chunk_id, source_key, crawl_root=ROOT, **meta):
    metadata = {"source_key": source_key, "crawl_root": crawl_root}
    metadata.update(meta)
    return TextChunk(id=chunk_id, text=f"body-{chunk_id}", metadata=metadata)


class _Store:
    def __init__(self, chunks=None, spec=None):
        self.chunks = list(chunks or [])
        self.spec = spec
        self.marked = []
        self.saved = []

    def get_crawl_spec(self, brain_id):
        return self.spec

    def save_crawl_spec(self, spec, brain_id):
        self.spec = spec
        self.saved.append(spec)
        return spec

    def delete_crawl_spec(self, brain_id):
        if self.spec is None:
            return False
        self.spec = None
        return True

    def get_text_chunks(
        self,
        brain_id,
        limit=10,
        skip=0,
        query_text=None,
        metadata_eq=None,
        order="desc",
    ):
        matched = []
        for chunk in self.chunks:
            meta = chunk.metadata or {}
            if metadata_eq and any(
                meta.get(key) != value for key, value in metadata_eq.items()
            ):
                continue
            matched.append(chunk)
        return matched[skip : skip + limit], len(matched)

    def mark_text_chunks_superseded(
        self, ids, superseded_at, brain_id, superseded_by=None, valid_to=None
    ):
        self.marked.append((list(ids), superseded_at, valid_to))
        wanted = set(ids)
        for chunk in self.chunks:
            if str(chunk.id) in wanted:
                chunk.metadata["superseded_at"] = superseded_at
                if valid_to:
                    chunk.metadata["valid_to"] = valid_to


class _Graph:
    def __init__(self, nodes=None, triples=None):
        self.nodes = list(nodes or [])
        self.triples = list(triples or [])
        self.updates = []

    def nodes_and_edges_citing(self, brain_id, chunk_ids, edge_uuids):
        wanted = set(chunk_ids)
        nodes = [
            node
            for node in self.nodes
            if wanted.intersection(
                (node.properties or {}).get("source_chunk_ids") or []
            )
        ]
        triples = []
        for triple in self.triples:
            props = triple[1].properties or {}
            if wanted.intersection(props.get("source_chunk_ids") or []):
                triples.append(triple)
        return nodes, triples

    def update_properties(
        self, fact_id, kind, brain_id=None, new_properties=None, **kwargs
    ):
        self.updates.append((fact_id, kind, dict(new_properties or {})))
        return True


class CrawlSpecRequestTests(unittest.TestCase):
    def test_defaults_match_url_ingest_bounds(self):
        body = CrawlSpecRequestBody(url="https://help.example.com/docs")
        self.assertEqual(body.max_pages, 10)
        self.assertEqual(body.interval_days, 7)
        self.assertTrue(body.enabled)
        self.assertEqual(body.allowed_path_prefixes, [])

    def test_rejects_http_and_out_of_range_pages(self):
        with self.assertRaises(ValidationError):
            CrawlSpecRequestBody(url="http://help.example.com/docs")
        with self.assertRaises(ValidationError):
            CrawlSpecRequestBody(url=ROOT, max_pages=0)
        with self.assertRaises(ValidationError):
            CrawlSpecRequestBody(url=ROOT, max_pages=1000)
        with self.assertRaises(ValidationError):
            CrawlSpecRequestBody(url=ROOT, interval_days=0)


class SourceKeyTests(unittest.TestCase):
    def test_page_source_key_is_stable_https(self):
        page = CrawledPage(
            url="https://Help.Example.com/docs/Refunds?lang=en",
            canonical_url="https://help.example.com/docs/refunds?lang=en",
            text="Refunds",
        )
        self.assertEqual(
            page_source_key(page),
            "https://help.example.com/docs/refunds?lang=en",
        )
        self.assertEqual(crawl_root_key(ROOT), ROOT)
        self.assertIsNone(normalize_source_key("http://help.example.com/docs"))


class DueClaimTests(unittest.TestCase):
    def test_new_spec_is_due_and_weekly_spec_is_not(self):
        self.assertTrue(spec_is_due(_spec(), now=NOW))
        self.assertFalse(
            spec_is_due(
                _spec(last_crawled_at="2026-11-24T12:00:00Z"),
                now=NOW,
            )
        )
        self.assertTrue(
            spec_is_due(
                _spec(last_crawled_at="2026-11-17T12:00:00Z"),
                now=NOW,
            )
        )
        self.assertFalse(spec_is_due(_spec(enabled=False), now=NOW, force=True))
        self.assertTrue(
            spec_is_due(
                _spec(last_crawled_at="2026-11-24T12:00:00Z"),
                now=NOW,
                force=True,
            )
        )

    def test_stale_in_flight_claim_becomes_due(self):
        started = (NOW - timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%SZ")
        self.assertTrue(
            spec_is_due(_spec(last_started_at=started), now=NOW)
        )
        recent = (NOW - timedelta(minutes=10)).strftime("%Y-%m-%dT%H:%M:%SZ")
        self.assertFalse(
            spec_is_due(_spec(last_started_at=recent), now=NOW)
        )

    def test_claim_fallback_marks_started(self):
        store = _Store(spec=_spec())
        claimed = claim_crawl_spec(store, "brain-a", now=NOW, task_id="task-1")
        self.assertEqual(claimed.last_task_id, "task-1")
        self.assertEqual(claimed.last_started_at, "2026-11-25T12:00:00Z")
        skipped = claim_crawl_spec(store, "brain-a", now=NOW)
        self.assertIsNone(skipped)


class CloseRemovedPagesTests(unittest.TestCase):
    def test_closes_missing_page_chunks_and_citing_facts(self):
        store = _Store(
            chunks=[
                _chunk("live-1", LIVE),
                _chunk("gone-1", GONE),
                _chunk(
                    "already",
                    "https://help.example.com/docs/archived",
                    superseded_at="2026-11-01T00:00:00Z",
                ),
            ]
        )
        graph = _Graph(
            nodes=[
                Node(
                    uuid="event-1",
                    name="Policy retired",
                    labels=["EVENT"],
                    properties={"source_chunk_ids": ["gone-1"]},
                ),
                Node(
                    uuid="person-1",
                    name="Ada",
                    labels=["PERSON"],
                    properties={"source_chunk_ids": ["gone-1"]},
                ),
            ],
            triples=[
                (
                    Node(uuid="org", name="Support", labels=["ORG"], properties={}),
                    Predicate(
                        uuid="rel-1",
                        name="STATES",
                        description="",
                        properties={"source_chunk_ids": ["gone-1"]},
                    ),
                    Node(uuid="clause", name="Refunds", labels=["CLAUSE"], properties={}),
                )
            ],
        )
        removed = close_removed_pages(
            store,
            graph,
            "brain-a",
            live_source_keys=[LIVE],
            crawl_root=ROOT,
            closed_at="2026-11-25T12:00:00Z",
        )
        self.assertEqual(removed, [GONE])
        self.assertEqual(store.marked[0][0], ["gone-1"])
        self.assertEqual(store.marked[0][2], "2026-11-25T12:00:00Z")
        kinds = {fact_id: kind for fact_id, kind, _ in graph.updates}
        self.assertEqual(kinds["rel-1"], "relationship")
        self.assertEqual(kinds["event-1"], "node")
        self.assertNotIn("person-1", kinds)

    def test_closes_previous_source_keys_from_another_root(self):
        old_root = "https://help.example.com/legacy"
        store = _Store(
            chunks=[_chunk("legacy-1", GONE, crawl_root=old_root)]
        )
        graph = _Graph()
        removed = close_removed_pages(
            store,
            graph,
            "brain-a",
            live_source_keys=[LIVE],
            crawl_root=ROOT,
            previous_source_keys=[GONE],
            closed_at="2026-11-25T12:00:00Z",
        )
        self.assertEqual(removed, [GONE])
        self.assertEqual(store.marked[0][0], ["legacy-1"])


class CrawlSpecRouteTests(unittest.TestCase):
    def _client(self):
        from src.services.api.dependencies import get_brain_id
        from src.services.api.routes.ingest import ingest_router

        app = FastAPI()
        app.dependency_overrides[get_brain_id] = lambda: "brain-a"
        app.include_router(ingest_router)
        return TestClient(app)

    def test_put_get_delete_round_trip(self):
        from src.services.api.controllers import crawl_spec as crawl_mod

        store = _Store()
        client = self._client()
        with patch.object(crawl_mod, "data_adapter", store):
            created = client.put(
                "/ingest/crawl-spec",
                json={
                    "url": ROOT,
                    "max_pages": 40,
                    "allowed_path_prefixes": ["/docs"],
                    "interval_days": 7,
                },
            )
            self.assertEqual(created.status_code, 200)
            self.assertEqual(created.json()["url"], ROOT)
            self.assertEqual(created.json()["cadence"], "scheduled re-crawl")
            self.assertEqual(created.json()["interval_days"], 7)
            fetched = client.get("/ingest/crawl-spec")
            self.assertEqual(fetched.status_code, 200)
            self.assertEqual(fetched.json()["max_pages"], 40)
            deleted = client.delete("/ingest/crawl-spec")
            self.assertEqual(deleted.status_code, 200)
            missing = client.get("/ingest/crawl-spec")
            self.assertEqual(missing.status_code, 404)

    def test_run_enqueues_forced_recrawl(self):
        from src.services.api.controllers import crawl_spec as crawl_mod

        store = _Store(spec=_spec())
        captured = {}

        def fake_apply_async(args=None, task_id=None):
            captured["args"] = args
            captured["task_id"] = task_id
            return MagicMock()

        client = self._client()
        with patch.object(crawl_mod, "data_adapter", store), patch(
            "src.services.api.controllers.crawl_spec.config"
        ) as config, patch(
            "src.workers.tasks.help_center.recrawl_help_center.apply_async",
            side_effect=fake_apply_async,
        ), patch(
            "src.workers.tasks.ingestion.set_ingestion_task_status",
            return_value={},
        ):
            config.crawler_is_configured = True
            response = client.post("/ingest/crawl-spec/run")
        self.assertEqual(response.status_code, 202)
        self.assertEqual(response.json()["message"], "Scheduled re-crawl accepted")
        self.assertEqual(captured["args"][0]["brain_id"], "brain-a")
        self.assertTrue(captured["args"][0]["force"])
        self.assertEqual(captured["task_id"], response.json()["task_id"])

    def test_run_without_crawler_is_503(self):
        from src.services.api.controllers import crawl_spec as crawl_mod

        apply_async = MagicMock()
        client = self._client()
        with patch.object(crawl_mod, "data_adapter", _Store(spec=_spec())), patch(
            "src.services.api.controllers.crawl_spec.config"
        ) as config, patch(
            "src.workers.tasks.help_center.recrawl_help_center.apply_async",
            apply_async,
        ):
            config.crawler_is_configured = False
            response = client.post("/ingest/crawl-spec/run")
        self.assertEqual(response.status_code, 503)
        apply_async.assert_not_called()

    def test_run_disabled_spec_is_409(self):
        from src.services.api.controllers import crawl_spec as crawl_mod

        client = self._client()
        with patch.object(
            crawl_mod, "data_adapter", _Store(spec=_spec(enabled=False))
        ), patch(
            "src.services.api.controllers.crawl_spec.config"
        ) as config:
            config.crawler_is_configured = True
            response = client.post("/ingest/crawl-spec/run")
        self.assertEqual(response.status_code, 409)


class RecrawlTaskTests(unittest.TestCase):
    def test_ingests_live_pages_closes_removed_and_scans_conflicts(self):
        from src.workers.tasks import help_center as help_mod

        request = MagicMock()
        request.id = "recrawl-1"
        store = _Store(
            spec=_spec(last_seen_source_keys=[LIVE, GONE]),
            chunks=[_chunk("gone-1", GONE)],
        )
        crawler = MagicMock()
        crawler.crawl_pages.return_value = [
            CrawledPage(url=LIVE, canonical_url=LIVE, text="Refunds stay 30 days."),
            CrawledPage(url=GONE, text="   "),
        ]
        enqueued = []
        scans = []

        def fake_ingest(args=None, task_id=None, **kwargs):
            enqueued.append({"args": args, "task_id": task_id})
            return MagicMock()

        def fake_scan(args=None, task_id=None, **kwargs):
            scans.append({"args": args, "task_id": task_id})
            return MagicMock()

        with patch.object(
            help_mod, "claim_crawl_spec", return_value=_spec(last_seen_source_keys=[LIVE, GONE])
        ), patch(
            "src.core.instances.crawler_adapter", crawler
        ), patch(
            "src.services.data.main.data_adapter", store
        ), patch(
            "src.services.kg_agent.main.graph_adapter", _Graph()
        ), patch(
            "src.workers.tasks.ingestion.ingest_data"
        ) as ingest_data, patch(
            "src.workers.tasks.conflicts.scan_conflicts.apply_async",
            side_effect=fake_scan,
        ), patch(
            "src.workers.tasks.ingestion.set_ingestion_task_status",
            return_value={},
        ):
            ingest_data.apply_async.side_effect = fake_ingest
            result = help_mod.recrawl_help_center.run.__func__(
                type("Bound", (), {"request": request})(),
                {"brain_id": "brain-a", "force": True},
            )

        self.assertEqual(result["pages"], 1)
        self.assertEqual(result["removed"], [GONE])
        payload = enqueued[0]["args"][0]
        self.assertEqual(payload["meta_keys"]["source_key"], LIVE)
        self.assertEqual(payload["meta_keys"]["crawl_root"], ROOT)
        self.assertEqual(scans[0]["args"][0]["brain_id"], "brain-a")
        self.assertFalse(scans[0]["args"][0]["use_llm"])
        self.assertTrue(result["last_crawled_at"])
        self.assertEqual(store.saved[-1].last_crawled_at, result["last_crawled_at"])
        self.assertEqual(store.saved[-1].pages_removed, 1)

    def test_skips_when_spec_is_not_due(self):
        from src.workers.tasks import help_center as help_mod

        request = MagicMock()
        request.id = "skip-1"
        crawler = MagicMock()
        with patch.object(help_mod, "claim_crawl_spec", return_value=None), patch(
            "src.core.instances.crawler_adapter", crawler
        ), patch(
            "src.workers.tasks.ingestion.set_ingestion_task_status",
            return_value={},
        ):
            result = help_mod.recrawl_help_center.run.__func__(
                type("Bound", (), {"request": request})(),
                {"brain_id": "brain-a"},
            )
        self.assertTrue(result["skipped"])
        crawler.crawl_pages.assert_not_called()

    def test_due_ticker_enqueues_each_brain(self):
        from src.workers.tasks import help_center as help_mod

        queued = []

        def fake_apply_async(args=None, **kwargs):
            queued.append(args[0]["brain_id"])
            return MagicMock()

        data = MagicMock()
        data.list_due_crawl_specs.return_value = [
            _spec(brain_id="brain-a"),
            _spec(brain_id="brain-b"),
        ]
        with patch("src.services.data.main.data_adapter", data), patch.object(
            help_mod.recrawl_help_center, "apply_async", side_effect=fake_apply_async
        ):
            result = help_mod.run_due_help_center_crawls.run.__func__(
                type("Bound", (), {"request": MagicMock()})(),
                {},
            )
        self.assertEqual(result["queued"], ["brain-a", "brain-b"])
        self.assertEqual(queued, ["brain-a", "brain-b"])


class HelpCenterQueueRoutingTests(unittest.TestCase):
    def test_help_center_tasks_route_to_ingest_data(self):
        from src.workers.app import ingestion_app

        for name in (
            "src.workers.tasks.help_center.recrawl_help_center",
            "src.workers.tasks.help_center.run_due_help_center_crawls",
        ):
            route = ingestion_app.amqp.router.route({}, name)
            self.assertEqual(route["queue"].name, "ingest_data")


class MarkFinishedTests(unittest.TestCase):
    def test_error_keeps_previous_last_crawled_at(self):
        spec = _spec(last_crawled_at="2026-11-18T12:00:00Z")
        failed = mark_finished(
            spec,
            live_source_keys=[LIVE],
            removed_source_keys=[],
            conflict_scan_task_id=None,
            now=NOW,
            error="crawler timeout",
        )
        self.assertEqual(failed.last_crawled_at, "2026-11-18T12:00:00Z")
        self.assertEqual(failed.last_error, "crawler timeout")


if __name__ == "__main__":
    unittest.main()
