from __future__ import annotations

import time
from typing import Any, Callable, Optional
from urllib.parse import urljoin

import httpx

from src.adapters.interfaces.crawler import CrawledPage, CrawlerClient, CrawlRequest
from src.lib.crawler.cleaned import cleaned_title, format_cleaned_page
from src.lib.crawler.errors import (
    CrawlerConfigError,
    CrawlerFailedError,
    CrawlerQuotaError,
    CrawlerTimeoutError,
)

TERMINAL_CRAWL_STATUSES = {"completed", "cancelled", "failed"}
TERMINAL_CLEANUP_STATUSES = {"completed", "failed", "skipped"}
POLL_INTERVAL_START = 1.0
POLL_INTERVAL_CAP = 5.0
PAGE_LIMIT = 100


class LumenCrawlerClient(CrawlerClient):
    def __init__(
        self,
        api_key: Optional[str] = None,
        endpoint: str = "https://crawler.lumen-labs.ai",
        poll_timeout_seconds: float = 900,
        timeout: float = 60.0,
        transport: Optional[httpx.BaseTransport] = None,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ):
        self.api_key = (api_key or "").strip()
        self.endpoint = endpoint.rstrip("/")
        self.poll_timeout_seconds = poll_timeout_seconds
        self.timeout = timeout
        self.transport = transport
        self._sleep = sleep
        self._monotonic = monotonic

    def _client(self) -> httpx.Client:
        return httpx.Client(timeout=self.timeout, transport=self.transport)

    def _headers(self, idempotency_key: Optional[str] = None) -> dict[str, str]:
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        if idempotency_key:
            headers["Idempotency-Key"] = idempotency_key
            headers["Content-Type"] = "application/json"
        return headers

    def _url(self, path: str) -> str:
        return urljoin(f"{self.endpoint}/", path.lstrip("/"))

    def crawl_pages(self, request: CrawlRequest) -> list[CrawledPage]:
        if not self.api_key:
            raise CrawlerConfigError("LUMEN_CRAWLER_KEY is not set")
        with self._client() as client:
            crawl_id = self._create_crawl(client, request)
            self._poll_until_cleaned(client, crawl_id)
            return self._list_cleaned_pages(client, crawl_id)

    def _create_crawl(self, client: httpx.Client, request: CrawlRequest) -> str:
        response = client.post(
            self._url("/v1/crawls"),
            headers=self._headers(request.idempotency_key),
            json={
                "url": request.url,
                "max_pages": request.max_pages,
                "allowed_domains": request.allowed_domains,
                "follow_doors": False,
                "cleanup": True,
            },
        )
        self._raise_for_status(response)
        body = response.json()
        crawl_id = body.get("id")
        if not crawl_id:
            raise CrawlerFailedError("Crawler did not return a crawl id")
        return str(crawl_id)

    def _poll_until_cleaned(self, client: httpx.Client, crawl_id: str) -> None:
        deadline = self._monotonic() + self.poll_timeout_seconds
        interval = POLL_INTERVAL_START
        while True:
            response = client.get(
                self._url(f"/v1/crawls/{crawl_id}"),
                headers=self._headers(),
            )
            self._raise_for_status(response)
            body = response.json()
            status = body.get("status")
            cleanup_status = (body.get("cleanup") or {}).get("status")
            if status in TERMINAL_CRAWL_STATUSES:
                if status != "completed":
                    reason = body.get("stop_reason") or status
                    raise CrawlerFailedError(f"Crawl ended with status {reason}")
                if cleanup_status == "completed":
                    return
                if cleanup_status in {"failed", "skipped"}:
                    raise CrawlerFailedError(
                        f"Crawl cleanup ended with status {cleanup_status}"
                    )
            if self._monotonic() >= deadline:
                self._cancel(client, crawl_id)
                raise CrawlerTimeoutError("Timed out waiting for crawl cleanup")
            self._sleep(interval)
            interval = min(interval * 1.5, POLL_INTERVAL_CAP)

    def _cancel(self, client: httpx.Client, crawl_id: str) -> None:
        try:
            client.post(
                self._url(f"/v1/crawls/{crawl_id}/cancel"),
                headers=self._headers(),
            )
        except httpx.HTTPError:
            return

    def _list_cleaned_pages(
        self, client: httpx.Client, crawl_id: str
    ) -> list[CrawledPage]:
        pages: list[CrawledPage] = []
        cursor: Optional[str] = None
        while True:
            params: dict[str, Any] = {"include": "cleaned", "limit": PAGE_LIMIT}
            if cursor:
                params["cursor"] = cursor
            response = client.get(
                self._url(f"/v1/crawls/{crawl_id}/pages"),
                headers=self._headers(),
                params=params,
            )
            self._raise_for_status(response)
            body = response.json()
            for item in body.get("items") or []:
                page = self._to_crawled_page(item)
                if page is not None:
                    pages.append(page)
            cursor = body.get("next_cursor")
            if not cursor:
                return pages

    def _to_crawled_page(self, item: dict) -> Optional[CrawledPage]:
        if not item.get("ok"):
            return None
        cleaned = item.get("cleaned")
        if not isinstance(cleaned, dict):
            return None
        text = format_cleaned_page(cleaned)
        if not text:
            return None
        url = cleaned.get("canonical_url") or cleaned.get("url") or item.get("url")
        if not url:
            return None
        return CrawledPage(
            url=str(url),
            canonical_url=cleaned.get("canonical_url") or item.get("canonical_url"),
            title=cleaned_title(cleaned, item.get("title")),
            page_type=cleaned.get("page_type"),
            text=text,
            content_hash=item.get("content_hash"),
            status_code=item.get("status_code"),
        )

    def _raise_for_status(self, response: httpx.Response) -> None:
        if response.is_success:
            return
        detail = _problem_detail(response)
        if response.status_code == 402:
            raise CrawlerQuotaError(detail)
        if response.status_code == 401:
            raise CrawlerConfigError(detail)
        raise CrawlerFailedError(detail)


def _problem_detail(response: httpx.Response) -> str:
    try:
        body = response.json()
    except ValueError:
        body = {}
    if isinstance(body, dict):
        detail = body.get("detail") or body.get("title") or body.get("type")
        if detail:
            return str(detail)
    return f"Crawler request failed with HTTP {response.status_code}"
