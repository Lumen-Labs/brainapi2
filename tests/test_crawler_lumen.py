from __future__ import annotations

import json

import httpx
import pytest

from src.adapters.crawler import CrawlerAdapter
from src.adapters.interfaces.crawler import CrawlRequest, CrawledPage
from src.lib.crawler.cleaned import format_cleaned_page
from src.lib.crawler.client_lumen import LumenCrawlerClient
from src.lib.crawler.errors import (
    CrawlerConfigError,
    CrawlerFailedError,
    CrawlerQuotaError,
    CrawlerTimeoutError,
)

CRAWL_ID = "crawl-1"
CLEANED_PAGE = {
    "url": "https://example.com/",
    "canonical_url": "https://example.com/",
    "page_type": "landing",
    "chrome": {"navigation": ["Home", "Pricing"], "footer": ["Login"], "forms": []},
    "content": {
        "title": "Home",
        "headings": [{"level": 1, "text": "Home"}, {"level": 2, "text": "Features"}],
        "core": "Welcome to Example.",
        "paragraphs": ["Welcome to Example.", "Home Pricing Login"],
    },
    "areas": [
        {
            "id": "main",
            "heading": "Home",
            "category": "intro",
            "markdown": "# Home\n\n" + ("huge markdown " * 200),
        }
    ],
    "facts": {
        "contacts": {
            "emails": ["hi@example.com"],
            "phones": [],
            "addresses": [],
        },
        "ctas": ["Sign up"],
    },
    "model": "mock",
    "input_tokens": 1,
    "output_tokens": 1,
}


def _request() -> CrawlRequest:
    return CrawlRequest(
        url="https://example.com",
        max_pages=10,
        allowed_domains=[],
        idempotency_key="task-1",
    )


class FakeClock:
    def __init__(self):
        self.now = 0.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def _json(request: httpx.Request, status: int, body: dict, headers: dict | None = None) -> httpx.Response:
    return httpx.Response(
        status,
        request=request,
        headers=headers or {"content-type": "application/json"},
        content=json.dumps(body).encode("utf-8"),
    )


def test_format_cleaned_page_keeps_core_fields_and_drops_chrome_markdown():
    text = format_cleaned_page(CLEANED_PAGE)
    assert "Title: Home" in text
    assert "Type: landing" in text
    assert "URL: https://example.com/" in text
    assert "H1 Home" in text
    assert "Welcome to Example." in text
    assert "email: hi@example.com" in text
    assert "Sign up" in text
    assert "Pricing" not in text
    assert "huge markdown" not in text
    assert "Home Pricing Login" not in text


def test_missing_api_key_raises_before_http():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(500, request=request)

    client = LumenCrawlerClient(
        api_key="",
        transport=httpx.MockTransport(handler),
    )
    with pytest.raises(CrawlerConfigError, match="LUMEN_CRAWLER_KEY"):
        client.crawl_pages(_request())
    assert calls == []


def test_create_poll_and_cleaned_pages():
    polls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if request.method == "POST" and path == "/v1/crawls":
            body = json.loads(request.content.decode("utf-8"))
            assert body["cleanup"] is True
            assert body["follow_doors"] is False
            assert body["max_pages"] == 10
            assert request.headers["idempotency-key"] == "task-1"
            return _json(
                request,
                202,
                {
                    "id": CRAWL_ID,
                    "status": "queued",
                    "cleanup": {"status": "pending"},
                },
            )
        if request.method == "GET" and path == f"/v1/crawls/{CRAWL_ID}":
            polls["n"] += 1
            if polls["n"] == 1:
                return _json(
                    request,
                    200,
                    {"id": CRAWL_ID, "status": "running", "cleanup": {"status": "pending"}},
                )
            return _json(
                request,
                200,
                {
                    "id": CRAWL_ID,
                    "status": "completed",
                    "cleanup": {"status": "completed"},
                },
            )
        if request.method == "GET" and path == f"/v1/crawls/{CRAWL_ID}/pages":
            assert request.url.params.get("include") == "cleaned"
            return _json(
                request,
                200,
                {
                    "items": [
                        {
                            "url": "https://example.com/",
                            "canonical_url": "https://example.com/",
                            "title": "Home",
                            "status_code": 200,
                            "ok": True,
                            "content_hash": "abc",
                            "cleaned": CLEANED_PAGE,
                        },
                        {
                            "url": "https://example.com/robots",
                            "ok": False,
                            "cleaned": CLEANED_PAGE,
                        },
                    ],
                    "next_cursor": None,
                },
            )
        return httpx.Response(404, request=request)

    clock = FakeClock()
    client = LumenCrawlerClient(
        api_key="lc_live_test",
        transport=httpx.MockTransport(handler),
        sleep=clock.sleep,
        monotonic=clock.monotonic,
    )
    pages = client.crawl_pages(_request())
    assert len(pages) == 1
    assert pages[0].url == "https://example.com/"
    assert pages[0].title == "Home"
    assert pages[0].page_type == "landing"
    assert "Welcome to Example." in pages[0].text
    assert "huge markdown" not in pages[0].text
    assert polls["n"] == 2


def test_quota_error_is_not_retried():
    def handler(request: httpx.Request) -> httpx.Response:
        return _json(
            request,
            402,
            {
                "type": "quota_exceeded",
                "title": "Quota exceeded",
                "status": 402,
                "detail": "max_pages does not fit remaining included pages",
            },
        )

    client = LumenCrawlerClient(
        api_key="lc_live_test",
        transport=httpx.MockTransport(handler),
    )
    with pytest.raises(CrawlerQuotaError, match="remaining included pages"):
        client.crawl_pages(_request())


def test_failed_crawl_raises():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return _json(request, 202, {"id": CRAWL_ID, "status": "queued"})
        return _json(
            request,
            200,
            {
                "id": CRAWL_ID,
                "status": "failed",
                "stop_reason": "seed_failed",
                "cleanup": {"status": "skipped"},
            },
        )

    client = LumenCrawlerClient(
        api_key="lc_live_test",
        transport=httpx.MockTransport(handler),
        sleep=lambda _: None,
    )
    with pytest.raises(CrawlerFailedError, match="seed_failed"):
        client.crawl_pages(_request())


def test_timeout_cancels_running_crawl():
    cancelled = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path.endswith("/cancel"):
            cancelled["n"] += 1
            return _json(request, 200, {"id": CRAWL_ID, "status": "cancelled"})
        if request.method == "POST":
            return _json(request, 202, {"id": CRAWL_ID, "status": "queued"})
        return _json(
            request,
            200,
            {"id": CRAWL_ID, "status": "running", "cleanup": {"status": "pending"}},
        )

    clock = FakeClock()
    client = LumenCrawlerClient(
        api_key="lc_live_test",
        transport=httpx.MockTransport(handler),
        poll_timeout_seconds=0,
        sleep=clock.sleep,
        monotonic=clock.monotonic,
    )
    with pytest.raises(CrawlerTimeoutError):
        client.crawl_pages(_request())
    assert cancelled["n"] == 1


def test_adapter_delegates_crawl_pages():
    adapter = CrawlerAdapter()
    page = CrawledPage(url="https://example.com/", text="Title: Home")

    class FakeClient:
        def crawl_pages(self, request: CrawlRequest) -> list[CrawledPage]:
            assert request.url == "https://example.com"
            return [page]

    adapter.add_client(FakeClient())
    result = adapter.crawl_pages(_request())
    assert result == [page]
