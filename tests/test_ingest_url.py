import os
import unittest
from unittest.mock import MagicMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from src.adapters.interfaces.crawler import CrawledPage
from src.lib.crawler.errors import CrawlerQuotaError, CrawlerTimeoutError
from src.services.api.constants.requests import IngestionUrlRequestBody


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


PAGE = CrawledPage(
    url="https://example.com/",
    canonical_url="https://example.com/",
    title="Home",
    page_type="landing",
    text="Title: Home\n\nWelcome to Example.",
    content_hash="abc",
    status_code=200,
)


class IngestionUrlRequestTests(unittest.TestCase):
    def test_accepts_https_and_defaults(self):
        body = IngestionUrlRequestBody(url="https://example.com/docs")
        self.assertEqual(body.max_pages, 10)
        self.assertEqual(body.allowed_domains, [])
        self.assertFalse(body.skip_enrichment)

    def test_rejects_http_and_invalid_bounds(self):
        with self.assertRaises(ValidationError):
            IngestionUrlRequestBody(url="http://example.com")
        with self.assertRaises(ValidationError):
            IngestionUrlRequestBody(url="https://example.com", max_pages=0)
        with self.assertRaises(ValidationError):
            IngestionUrlRequestBody(url="https://example.com", max_pages=101)
        with self.assertRaises(ValidationError):
            IngestionUrlRequestBody(
                url="https://example.com",
                allowed_domains=[f"host{i}.com" for i in range(21)],
            )


class IngestUrlRouteTests(unittest.TestCase):
    def _client(self):
        from src.services.api.dependencies import get_brain_id
        from src.services.api.routes.ingest import ingest_router

        app = FastAPI()
        app.dependency_overrides[get_brain_id] = lambda: "brain-a"
        app.include_router(ingest_router)
        return TestClient(app)

    def test_valid_request_enqueues_task(self):
        captured = {}

        def fake_apply_async(args=None, task_id=None):
            captured["args"] = args
            captured["task_id"] = task_id
            return MagicMock()

        client = self._client()
        with patch(
            "src.services.api.routes.ingest.config"
        ) as config, patch(
            "src.services.api.routes.ingest.set_ingestion_task_status",
            return_value={},
        ), patch(
            "src.services.api.routes.ingest.ingest_url_task.apply_async",
            side_effect=fake_apply_async,
        ):
            config.crawler_is_configured = True
            response = client.post(
                "/ingest/url",
                json={"url": "https://example.com", "max_pages": 3},
            )
        self.assertEqual(response.status_code, 202)
        self.assertEqual(response.json()["message"], "URL ingestion accepted")
        self.assertTrue(response.json()["task_id"])
        self.assertEqual(captured["args"][0]["brain_id"], "brain-a")
        self.assertEqual(captured["args"][0]["url"], "https://example.com")
        self.assertEqual(captured["args"][0]["max_pages"], 3)

    def test_missing_crawler_key_is_503_without_enqueue(self):
        apply_async = MagicMock()
        client = self._client()
        with patch(
            "src.services.api.routes.ingest.config"
        ) as config, patch(
            "src.services.api.routes.ingest.ingest_url_task.apply_async",
            apply_async,
        ):
            config.crawler_is_configured = False
            response = client.post(
                "/ingest/url",
                json={"url": "https://example.com"},
            )
        self.assertEqual(response.status_code, 503)
        apply_async.assert_not_called()

    def test_invalid_url_is_422(self):
        client = self._client()
        with patch(
            "src.services.api.routes.ingest.config"
        ) as config:
            config.crawler_is_configured = True
            response = client.post(
                "/ingest/url",
                json={"url": "http://localhost"},
            )
        self.assertEqual(response.status_code, 422)


class IngestUrlTaskTests(unittest.TestCase):
    def test_fans_out_cleaned_pages_with_source_url(self):
        from src.workers.tasks import ingestion as ingestion_mod

        request = MagicMock()
        request.id = "url-parent"
        enqueued = []
        statuses = []

        def fake_apply_async(args=None, task_id=None):
            enqueued.append({"args": args, "task_id": task_id})
            return MagicMock()

        def fake_set_status(task_id, brain_id, status, **kwargs):
            statuses.append((task_id, status, kwargs.get("stage")))
            return {"status": status, "task_id": task_id}

        crawler = MagicMock()
        crawler.crawl_pages.return_value = [
            PAGE,
            CrawledPage(url="https://example.com/empty", text="   "),
        ]

        with patch(
            "src.core.instances.crawler_adapter", crawler
        ), patch.object(
            ingestion_mod, "ingest_data"
        ) as ingest_data, patch.object(
            ingestion_mod, "set_ingestion_task_status", side_effect=fake_set_status
        ):
            ingest_data.apply_async.side_effect = fake_apply_async
            result = ingestion_mod.ingest_url.run.__func__(
                type("Bound", (), {"request": request})(),
                {
                    "url": "https://example.com",
                    "max_pages": 10,
                    "brain_id": "brain-a",
                    "skip_enrichment": True,
                },
            )

        self.assertEqual(result["task_id"], "url-parent")
        self.assertEqual(len(result["task_ids"]), 1)
        payload = enqueued[0]["args"][0]
        self.assertEqual(payload["meta_keys"]["source_url"], "https://example.com/")
        self.assertEqual(payload["meta_keys"]["source_title"], "Home")
        self.assertEqual(payload["skip_enrichment"], True)
        self.assertIn("Welcome to Example.", payload["data"]["text_data"])
        self.assertEqual(statuses[-1][:2], ("url-parent", "completed"))
        crawl_request = crawler.crawl_pages.call_args.args[0]
        self.assertEqual(crawl_request.idempotency_key, "url-parent")

    def test_empty_crawl_completes_with_zero_pages(self):
        from src.workers.tasks import ingestion as ingestion_mod

        request = MagicMock()
        request.id = "url-empty"
        crawler = MagicMock()
        crawler.crawl_pages.return_value = []

        with patch(
            "src.core.instances.crawler_adapter", crawler
        ), patch.object(
            ingestion_mod, "ingest_data"
        ) as ingest_data, patch.object(
            ingestion_mod, "set_ingestion_task_status", return_value={}
        ):
            result = ingestion_mod.ingest_url.run.__func__(
                type("Bound", (), {"request": request})(),
                {"url": "https://example.com", "brain_id": "brain-a"},
            )

        self.assertEqual(result["task_ids"], [])
        ingest_data.apply_async.assert_not_called()

    def test_quota_failure_marks_parent_failed(self):
        from src.workers.tasks import ingestion as ingestion_mod

        request = MagicMock()
        request.id = "url-quota"
        statuses = []
        crawler = MagicMock()
        crawler.crawl_pages.side_effect = CrawlerQuotaError("max_pages does not fit")

        def fake_set_status(task_id, brain_id, status, **kwargs):
            statuses.append((status, kwargs.get("error")))
            return {"status": status, "task_id": task_id}

        with patch(
            "src.core.instances.crawler_adapter", crawler
        ), patch.object(
            ingestion_mod, "set_ingestion_task_status", side_effect=fake_set_status
        ):
            with self.assertRaises(CrawlerQuotaError):
                ingestion_mod.ingest_url.run.__func__(
                    type("Bound", (), {"request": request})(),
                    {"url": "https://example.com", "brain_id": "brain-a"},
                )

        self.assertEqual(statuses[-1][0], "failed")
        self.assertIn("max_pages", statuses[-1][1])

    def test_timeout_failure_marks_parent_failed(self):
        from src.workers.tasks import ingestion as ingestion_mod

        request = MagicMock()
        request.id = "url-timeout"
        statuses = []
        crawler = MagicMock()
        crawler.crawl_pages.side_effect = CrawlerTimeoutError("Timed out waiting")

        def fake_set_status(task_id, brain_id, status, **kwargs):
            statuses.append(status)
            return {"status": status, "task_id": task_id}

        with patch(
            "src.core.instances.crawler_adapter", crawler
        ), patch.object(
            ingestion_mod, "set_ingestion_task_status", side_effect=fake_set_status
        ):
            with self.assertRaises(CrawlerTimeoutError):
                ingestion_mod.ingest_url.run.__func__(
                    type("Bound", (), {"request": request})(),
                    {"url": "https://example.com", "brain_id": "brain-a"},
                )

        self.assertEqual(statuses[-1], "failed")


if __name__ == "__main__":
    unittest.main()
