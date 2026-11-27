from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit

from datetime import datetime, timezone

from src.core.saving.validity import format_instant, parse_instant, utc_now

DEFAULT_WEBHOOK_MAX_AGE_SECONDS = 300
WEBHOOK_FUTURE_SKEW_SECONDS = 60
DEFAULT_LIST_PAGE_SIZE = 100
ZENDESK_MAX_PAGE_SIZE = 100
DEFAULT_LIST_MAX_PAGES = 50

HttpGet = Callable[[str, dict[str, str]], dict[str, Any]]


class WebhookRejected(ValueError):
    pass


class SourceFetchError(RuntimeError):
    pass


@dataclass(frozen=True)
class SourceEvent:
    action: str
    document_id: str
    revision: str
    title: str = ""
    body: str = ""
    url: str = ""
    occurred_at: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class SourceConnector(Protocol):
    source_type: str
    supports_webhooks: bool

    def verify_webhook(
        self,
        headers: dict[str, str],
        body: bytes,
        secret: str,
    ) -> bool: ...

    def parse_webhook(self, headers: dict[str, str], body: bytes) -> list[SourceEvent]: ...

    def list_documents(self, connection: dict[str, Any]) -> list[SourceEvent]: ...

    def fetch_document(
        self, connection: dict[str, Any], document_id: str
    ) -> SourceEvent | None: ...


def _headers_map(headers: dict[str, str] | None) -> dict[str, str]:
    return {str(key).lower(): str(value) for key, value in (headers or {}).items()}


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _revision_from(detail: dict[str, Any], body: str) -> str:
    for key in ("updated_at", "edited_at", "revision_id", "html_url"):
        text = _text(detail.get(key))
        if text:
            return text
    digest = hashlib.sha256((body or _text(detail.get("title"))).encode("utf-8")).hexdigest()
    return digest


def _occurred_at(detail: dict[str, Any]) -> str:
    raw = detail.get("updated_at") or detail.get("created_at")
    parsed = parse_instant(raw)
    return format_instant(parsed or utc_now())


def _positive_env(name: str, default: int, *, ceiling: int | None = None) -> int:
    raw = os.getenv(name, str(default))
    try:
        parsed = int(raw)
    except (TypeError, ValueError):
        parsed = default
    if parsed <= 0:
        parsed = default
    if ceiling is not None:
        parsed = min(parsed, ceiling)
    return parsed


def webhook_max_age_seconds() -> int:
    return _positive_env(
        "SOURCE_SYNC_WEBHOOK_MAX_AGE_SECONDS", DEFAULT_WEBHOOK_MAX_AGE_SECONDS
    )


def list_page_size() -> int:
    return _positive_env(
        "SOURCE_SYNC_LIST_PAGE_SIZE",
        DEFAULT_LIST_PAGE_SIZE,
        ceiling=ZENDESK_MAX_PAGE_SIZE,
    )


def list_max_pages() -> int:
    return _positive_env("SOURCE_SYNC_LIST_MAX_PAGES", DEFAULT_LIST_MAX_PAGES)


def _url_with_query(url: str, **params: Any) -> str:
    parts = urlsplit(url)
    query = dict(parse_qsl(parts.query, keep_blank_values=True))
    for key, value in params.items():
        if value is None:
            continue
        query[str(key)] = str(value)
    return urlunsplit(
        (parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment)
    )


def _articles_batch(page: Any) -> list[Any]:
    if not isinstance(page, dict):
        raise SourceFetchError("article listing returned a malformed page")
    batch = page.get("articles")
    if not isinstance(batch, list):
        raise SourceFetchError("article listing page missing articles list")
    return batch


def _next_page_url(page: dict[str, Any]) -> str:
    next_page = page.get("next_page")
    if isinstance(next_page, dict):
        href = _text(next_page.get("href"))
        if href:
            return href
    url = _text(next_page)
    if url:
        return url
    links = _as_dict(page.get("links"))
    return _text(links.get("next"))


def generic_hmac_hex(secret: str, body: bytes, timestamp: str = "") -> str:
    signed = f"{timestamp}.".encode("utf-8") + body if timestamp else body
    return hmac.new(secret.encode("utf-8"), signed, hashlib.sha256).hexdigest()


def webhook_timestamp_fresh(
    timestamp: str,
    *,
    now: datetime | None = None,
    max_age_seconds: int | None = None,
) -> bool:
    parsed = parse_instant(timestamp)
    if parsed is None:
        try:
            parsed = datetime.fromtimestamp(float(timestamp), tz=timezone.utc)
        except (TypeError, ValueError, OverflowError, OSError):
            return False
    current = now or utc_now()
    delta = (current - parsed).total_seconds()
    if delta < -WEBHOOK_FUTURE_SKEW_SECONDS:
        return False
    age = webhook_max_age_seconds() if max_age_seconds is None else int(max_age_seconds)
    return delta <= age


def zendesk_signature(secret: str, timestamp: str, body: bytes) -> str:
    signed = timestamp.encode("utf-8") + body
    digest = hmac.new(secret.encode("utf-8"), signed, hashlib.sha256).digest()
    return base64.b64encode(digest).decode("ascii")


class ZendeskGuideConnector:
    source_type = "zendesk_guide"
    supports_webhooks = True

    def __init__(self, http_get: HttpGet | None = None):
        self._http_get = http_get

    def verify_webhook(self, headers: dict[str, str], body: bytes, secret: str) -> bool:
        if not secret:
            return False
        mapped = _headers_map(headers)
        generic = mapped.get("x-source-sync-signature")
        timestamp = mapped.get("x-source-sync-timestamp") or mapped.get(
            "x-zendesk-webhook-signature-timestamp", ""
        )
        if not timestamp or not webhook_timestamp_fresh(timestamp):
            return False
        if generic:
            expected = generic_hmac_hex(secret, body, timestamp)
            return hmac.compare_digest(generic, expected)
        signature = mapped.get("x-zendesk-webhook-signature", "")
        if not signature:
            return False
        expected = zendesk_signature(secret, timestamp, body)
        return hmac.compare_digest(signature, expected)

    def parse_webhook(self, headers: dict[str, str], body: bytes) -> list[SourceEvent]:
        try:
            payload = json.loads(body.decode("utf-8") or "{}")
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise WebhookRejected("webhook body must be JSON") from exc
        if not isinstance(payload, dict):
            raise WebhookRejected("webhook body must be a JSON object")
        events = payload.get("events")
        if isinstance(events, list):
            parsed = []
            for item in events:
                event = self._event_from_payload(item if isinstance(item, dict) else {})
                if event is not None:
                    parsed.append(event)
            return parsed
        event = self._event_from_payload(payload)
        return [event] if event is not None else []

    def list_documents(self, connection: dict[str, Any]) -> list[SourceEvent]:
        articles = []
        url = _url_with_query(self._articles_url(connection), per_page=list_page_size())
        headers = self._auth_headers(connection)
        getter = self._getter()
        limit = list_max_pages()
        seen = 0
        while url:
            if seen >= limit:
                raise SourceFetchError(
                    f"article listing exceeded {limit} pages without completing"
                )
            page = getter(url, headers)
            batch = _articles_batch(page)
            for article in batch:
                event = self._event_from_article(_as_dict(article), action="upsert")
                if event is not None:
                    articles.append(event)
            url = _next_page_url(page)
            seen += 1
        return articles

    def fetch_document(
        self, connection: dict[str, Any], document_id: str
    ) -> SourceEvent | None:
        article_id = _text(document_id)
        if not article_id:
            return None
        url = urljoin(self._api_root(connection), f"help_center/articles/{article_id}.json")
        page = self._getter()(url, self._auth_headers(connection))
        article = _as_dict(page.get("article") if isinstance(page, dict) else page)
        return self._event_from_article(article, action="upsert")

    def _event_from_payload(self, payload: dict[str, Any]) -> SourceEvent | None:
        event_type = _text(
            payload.get("type")
            or payload.get("event_type")
            or _as_dict(payload.get("event")).get("type")
        ).lower()
        detail = _as_dict(payload.get("detail") or payload.get("article") or payload)
        if "article" in _as_dict(payload.get("event")):
            nested = _as_dict(_as_dict(payload.get("event")).get("article"))
            detail = {**nested, **detail} if detail else nested
        action = self._action_for(event_type, detail)
        if action is None:
            return None
        return self._event_from_article(detail, action=action)

    def _event_from_article(
        self, article: dict[str, Any], *, action: str
    ) -> SourceEvent | None:
        document_id = _text(article.get("id") or article.get("article_id"))
        if not document_id:
            return None
        body = _text(article.get("body") or article.get("body_html") or article.get("text"))
        title = _text(article.get("title") or article.get("name"))
        url = _text(article.get("html_url") or article.get("url"))
        revision = _revision_from(article, body)
        draft = article.get("draft")
        if action == "upsert" and draft is True:
            action = "delete"
        return SourceEvent(
            action=action,
            document_id=document_id,
            revision=revision,
            title=title,
            body=body,
            url=url,
            occurred_at=_occurred_at(article),
            metadata={"locale": _text(article.get("locale"))},
        )

    @staticmethod
    def _action_for(event_type: str, detail: dict[str, Any]) -> str | None:
        if "unpublished" in event_type or "deleted" in event_type:
            return "delete"
        if "published" in event_type or "updated" in event_type or "created" in event_type:
            return "upsert"
        if detail.get("id") or detail.get("article_id"):
            return "upsert"
        return None

    def _articles_url(self, connection: dict[str, Any]) -> str:
        locale = _text(connection.get("locale"))
        if locale:
            return urljoin(
                self._api_root(connection),
                f"help_center/{locale}/articles.json",
            )
        return urljoin(self._api_root(connection), "help_center/articles.json")

    def _api_root(self, connection: dict[str, Any]) -> str:
        base = _text(connection.get("base_url")).rstrip("/") + "/"
        if not base.strip("/"):
            raise SourceFetchError("base_url is required for zendesk_guide poll")
        if base.rstrip("/").endswith("/api/v2"):
            return base if base.endswith("/") else base + "/"
        return urljoin(base, "api/v2/")

    def _auth_headers(self, connection: dict[str, Any]) -> dict[str, str]:
        token = _text(os.getenv(_text(connection.get("api_token_env")), ""))
        email = _text(connection.get("api_email") or os.getenv("ZENDESK_GUIDE_API_EMAIL", ""))
        if not token:
            raise SourceFetchError("api_token_env is missing or empty")
        if email:
            raw = f"{email}/token:{token}".encode("utf-8")
            encoded = base64.b64encode(raw).decode("ascii")
            return {"Authorization": f"Basic {encoded}"}
        return {"Authorization": f"Bearer {token}"}

    def _getter(self) -> HttpGet:
        if self._http_get is not None:
            return self._http_get
        return _default_http_get


CONNECTORS: dict[str, SourceConnector] = {
    ZendeskGuideConnector.source_type: ZendeskGuideConnector(),
}


def get_connector(source_type: str) -> SourceConnector:
    connector = CONNECTORS.get((source_type or "").strip())
    if connector is None:
        raise WebhookRejected(f"unknown source type: {source_type}")
    return connector


def _default_http_get(url: str, headers: dict[str, str]) -> dict[str, Any]:
    import urllib.error
    import urllib.request

    request = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8") or "{}")
    except urllib.error.URLError as exc:
        raise SourceFetchError("source fetch failed") from exc
    if not isinstance(payload, dict):
        raise SourceFetchError("source fetch returned non-object JSON")
    return payload
