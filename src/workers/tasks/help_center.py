from __future__ import annotations

import logging
from uuid import uuid4

from src.core.saving.help_center_crawl import (
    RecheckQuotaStop,
    claim_crawl_spec,
    classify_crawl_pages,
    close_removed_pages,
    enqueue_conflict_scan,
    finish_crawl_spec,
    recheck_unseen_previous_pages,
    retained_source_keys,
    retirement_abort_reason,
)

logger = logging.getLogger(__name__)
from src.core.saving.ingest_access_gate import guard_worker
from src.core.saving.validity import format_instant, utc_now
from src.lib.tracing.ingest_telemetry import INGEST_URL_OPERATION, now_ms, telemetry_extra
from src.workers.app import ingestion_app


@ingestion_app.task(
    bind=True,
    name="src.workers.tasks.help_center.recrawl_help_center",
    acks_late=True,
    reject_on_worker_lost=True,
)
@guard_worker(lambda args: (args or {}).get("brain_id", "default"))
def recrawl_help_center(self, args: dict | None = None):
    from celery.exceptions import OperationalError

    from src.adapters.interfaces.crawler import CrawlRequest
    from src.config import config
    from src.core.instances import crawler_adapter
    from src.lib.crawler.errors import (
        CrawlerConfigError,
        CrawlerFailedError,
        CrawlerQuotaError,
        CrawlerTimeoutError,
    )
    from src.services.data.main import data_adapter
    from src.services.kg_agent.main import graph_adapter
    from src.workers.tasks.ingestion import (
        FILE_INGEST_MAX_RETRIES,
        FILE_INGEST_RETRY_DELAY,
        build_url_page_ingest_payload,
        ingest_data,
        set_ingestion_task_status,
    )

    payload = args or {}
    brain_id = payload.get("brain_id") or "default"
    force = bool(payload.get("force"))
    spec = claim_crawl_spec(
        data_adapter,
        brain_id,
        force=force,
        task_id=self.request.id,
    )
    if spec is None:
        set_ingestion_task_status(
            self.request.id,
            brain_id,
            "completed",
            stage="skipped",
            extra={"reason": "not_due"},
        )
        return {"brain_id": brain_id, "skipped": True}

    started_at_ms = now_ms()
    group_tracking = spec.group_tracking
    set_ingestion_task_status(
        self.request.id,
        brain_id,
        "started",
        stage="crawling",
        group_tracking=group_tracking,
        extra=telemetry_extra(
            operation=INGEST_URL_OPERATION,
            kind="help_center",
            pipeline_mode=config.pipeline_mode,
            started_at_ms=started_at_ms,
        ),
    )
    try:
        pages = crawler_adapter.crawl_pages(
            CrawlRequest(
                url=spec.url,
                max_pages=int(spec.max_pages or 10),
                allowed_domains=list(spec.allowed_domains or []),
                allowed_path_prefixes=list(spec.allowed_path_prefixes or []),
                idempotency_key=self.request.id,
            )
        )
        classified = recheck_unseen_previous_pages(
            classify_crawl_pages(pages, spec),
            spec,
            lambda url: _recheck_page(
                crawler_adapter,
                spec,
                url,
                self.request.id,
                (
                    CrawlerConfigError,
                    CrawlerQuotaError,
                    CrawlerFailedError,
                    CrawlerTimeoutError,
                    OSError,
                ),
            ),
            budget=getattr(config, "help_center_recheck_max_pages", 50),
        )
        task_ids = []
        for page in classified["ingestible"]:
            page_task_id = str(uuid4())
            task_ids.append(page_task_id)
            set_ingestion_task_status(
                page_task_id,
                brain_id,
                "queued",
                stage="queued",
                group_tracking=group_tracking,
            )
            child = build_url_page_ingest_payload(
                brain_id,
                page,
                group_tracking=group_tracking,
                skip_enrichment=bool(spec.skip_enrichment),
                crawl_root=spec.url,
            )
            _enqueue_page(
                ingest_data,
                child,
                page_task_id,
                self,
                OperationalError,
                FILE_INGEST_MAX_RETRIES,
                FILE_INGEST_RETRY_DELAY,
            )
        abort_reason = retirement_abort_reason(
            classified,
            spec,
            ratio=getattr(config, "help_center_retire_max_ratio", None),
            max_pages=getattr(config, "help_center_retire_max_pages", None),
        )
        closed_at = format_instant(utc_now())
        removed = []
        if abort_reason is None:
            removed = close_removed_pages(
                data_adapter,
                graph_adapter,
                brain_id,
                gone_source_keys=classified["gone_keys"],
                crawl_root=spec.url,
                closed_at=closed_at,
            )
        else:
            logger.warning(
                "help_center retirement skipped brain_id=%s reason=%s",
                brain_id,
                abort_reason,
            )
        recheck_warning = classified.get("recheck_stopped")
        warning = abort_reason
        if recheck_warning:
            warning = f"{warning}; {recheck_warning}" if warning else str(recheck_warning)
        finish_live = retained_source_keys(
            spec, classified, removed, aborted=abort_reason is not None
        )
        scan_id = enqueue_conflict_scan(brain_id)
        finished = finish_crawl_spec(
            data_adapter,
            brain_id,
            live_source_keys=finish_live,
            removed_source_keys=removed,
            conflict_scan_task_id=scan_id,
            task_id=self.request.id,
            warning=warning,
        )
        set_ingestion_task_status(
            self.request.id,
            brain_id,
            "completed",
            stage="completed",
            counts={
                "pages": len(task_ids),
                "task_ids": task_ids,
                "removed": len(removed),
                "failed": len(classified["failed_keys"]),
                "gone": len(classified["gone_keys"]),
            },
            group_tracking=group_tracking,
            extra=telemetry_extra(
                operation=INGEST_URL_OPERATION,
                kind="help_center",
                pipeline_mode=config.pipeline_mode,
                started_at_ms=started_at_ms,
                size_pages=len(task_ids),
                unit_count=len(task_ids),
            ),
        )
        return {
            "brain_id": brain_id,
            "task_id": self.request.id,
            "task_ids": task_ids,
            "pages": len(task_ids),
            "removed": removed,
            "conflict_scan_task_id": scan_id,
            "last_crawled_at": None if finished is None else finished.last_crawled_at,
            "retirement_aborted": abort_reason,
            "recheck_stopped": recheck_warning,
            "recheck_remaining": classified.get("recheck_remaining") or [],
        }
    except (
        CrawlerConfigError,
        CrawlerQuotaError,
        CrawlerFailedError,
        CrawlerTimeoutError,
        Exception,
    ) as exc:
        finish_crawl_spec(
            data_adapter,
            brain_id,
            live_source_keys=spec.last_seen_source_keys,
            removed_source_keys=[],
            conflict_scan_task_id=spec.last_conflict_scan_task_id,
            task_id=self.request.id,
            error=str(exc),
        )
        set_ingestion_task_status(
            self.request.id,
            brain_id,
            "failed",
            stage="failed",
            error=str(exc),
            group_tracking=group_tracking,
        )
        raise


@ingestion_app.task(
    bind=True,
    name="src.workers.tasks.help_center.run_due_help_center_crawls",
    acks_late=True,
    reject_on_worker_lost=True,
)
def run_due_help_center_crawls(self, args: dict | None = None):
    from src.services.data.main import data_adapter

    lister = getattr(data_adapter, "list_due_crawl_specs", None)
    specs = list(lister() if callable(lister) else [])
    queued = []
    for spec in specs:
        brain_id = getattr(spec, "brain_id", None)
        if not brain_id:
            continue
        recrawl_help_center.apply_async(args=[{"brain_id": brain_id, "force": False}])
        queued.append(brain_id)
    return {"queued": queued}


def _recheck_page(crawler, spec, url, task_id, errors):
    from src.adapters.interfaces.crawler import CrawlRequest
    from src.lib.crawler.errors import CrawlerQuotaError

    try:
        pages = list(
            crawler.crawl_pages(
                CrawlRequest(
                    url=url,
                    max_pages=1,
                    allowed_domains=list(spec.allowed_domains or []),
                    allowed_path_prefixes=list(spec.allowed_path_prefixes or []),
                    idempotency_key=f"{task_id}:recheck:{url}",
                )
            )
            or []
        )
    except CrawlerQuotaError as exc:
        raise RecheckQuotaStop("crawler quota exceeded") from exc
    except errors as exc:
        if _is_rate_or_quota(exc):
            raise RecheckQuotaStop("crawler rate limited") from exc
        return []
    if any(_page_is_rate_or_quota(page) for page in pages):
        raise RecheckQuotaStop("crawler rate limited")
    return pages


def _page_is_rate_or_quota(page) -> bool:
    try:
        status = int(getattr(page, "status_code", None) or 0)
    except (TypeError, ValueError):
        status = 0
    return status in {429, 402}


def _is_rate_or_quota(exc) -> bool:
    from src.lib.crawler.errors import CrawlerQuotaError

    if isinstance(exc, CrawlerQuotaError):
        return True
    try:
        status = int(getattr(exc, "status_code", None) or 0)
    except (TypeError, ValueError):
        status = 0
    if status in {429, 402}:
        return True
    text = str(exc).lower()
    return "429" in text or "rate limit" in text or "quota" in text


def _enqueue_page(
    ingest_data,
    payload,
    page_task_id,
    parent,
    operational_error,
    max_retries,
    retry_delay,
):
    import time

    for attempt in range(max_retries):
        try:
            ingest_data.apply_async(
                args=[payload],
                task_id=page_task_id,
                **_child_headers(parent, page_task_id),
            )
            return
        except operational_error:
            if attempt == max_retries - 1:
                raise
            time.sleep(retry_delay * (attempt + 1))


def _child_headers(parent, page_task_id):
    from src.core.saving.pipeline_runtime import child_pipeline_headers

    try:
        return child_pipeline_headers(parent, page_task_id)
    except Exception:
        return {}
