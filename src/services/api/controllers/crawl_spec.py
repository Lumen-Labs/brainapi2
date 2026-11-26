from __future__ import annotations

import asyncio
from uuid import uuid4

from fastapi import HTTPException
from starlette.responses import JSONResponse

from src.config import config
from src.core.saving.help_center_crawl import CrawlSpec, spec_in_flight, spec_is_due
from src.services.api.constants.requests import CrawlSpecRequestBody
from src.services.data.main import data_adapter


def _to_response(spec: CrawlSpec, message: str) -> dict:
    return {
        "message": message,
        "brain_id": spec.brain_id,
        "url": spec.url,
        "max_pages": spec.max_pages,
        "allowed_domains": list(spec.allowed_domains or []),
        "allowed_path_prefixes": list(spec.allowed_path_prefixes or []),
        "interval_days": spec.interval_days,
        "enabled": spec.enabled,
        "skip_enrichment": spec.skip_enrichment,
        "group_tracking": spec.group_tracking,
        "last_crawled_at": spec.last_crawled_at,
        "last_started_at": spec.last_started_at,
        "last_seen_source_keys": list(spec.last_seen_source_keys or []),
        "last_task_id": spec.last_task_id,
        "last_error": spec.last_error,
        "last_conflict_scan_task_id": spec.last_conflict_scan_task_id,
        "pages_ingested": spec.pages_ingested,
        "pages_removed": spec.pages_removed,
        "cadence": "scheduled re-crawl",
    }


def _merge_spec(brain_id: str, body: CrawlSpecRequestBody, previous: CrawlSpec | None) -> CrawlSpec:
    payload = body.model_dump()
    payload["brain_id"] = brain_id
    if previous is not None:
        payload["last_crawled_at"] = previous.last_crawled_at
        payload["last_started_at"] = previous.last_started_at
        payload["last_seen_source_keys"] = previous.last_seen_source_keys
        payload["last_task_id"] = previous.last_task_id
        payload["last_error"] = previous.last_error
        payload["last_conflict_scan_task_id"] = previous.last_conflict_scan_task_id
        payload["pages_ingested"] = previous.pages_ingested
        payload["pages_removed"] = previous.pages_removed
    return CrawlSpec.model_validate(payload)


async def put_crawl_spec(brain_id: str, body: CrawlSpecRequestBody):
    def _save():
        previous = data_adapter.get_crawl_spec(brain_id)
        spec = _merge_spec(brain_id, body, previous)
        return data_adapter.save_crawl_spec(spec, brain_id)

    saved = await asyncio.to_thread(_save)
    return _to_response(saved, "Crawl spec saved")


async def get_crawl_spec(brain_id: str):
    spec = await asyncio.to_thread(data_adapter.get_crawl_spec, brain_id)
    if spec is None:
        raise HTTPException(status_code=404, detail="No crawl spec is stored for this brain")
    return _to_response(spec, "Crawl spec")


async def delete_crawl_spec(brain_id: str):
    deleted = await asyncio.to_thread(data_adapter.delete_crawl_spec, brain_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="No crawl spec is stored for this brain")
    return {"message": "Crawl spec removed", "brain_id": brain_id}


async def run_crawl_spec(brain_id: str):
    from src.workers.tasks.help_center import recrawl_help_center
    from src.workers.tasks.ingestion import set_ingestion_task_status

    if not config.crawler_is_configured:
        raise HTTPException(status_code=503, detail="URL crawler is not configured")

    spec = await asyncio.to_thread(data_adapter.get_crawl_spec, brain_id)
    if spec is None:
        raise HTTPException(status_code=404, detail="No crawl spec is stored for this brain")
    if not spec.enabled:
        raise HTTPException(status_code=409, detail="Crawl spec is disabled")
    if spec_in_flight(spec) or not spec_is_due(spec, force=True):
        raise HTTPException(status_code=409, detail="A crawl is already running for this spec")

    task_id = str(uuid4())

    def _enqueue():
        set_ingestion_task_status(task_id, brain_id, "queued", stage="queued")
        recrawl_help_center.apply_async(
            args=[{"brain_id": brain_id, "force": True}],
            task_id=task_id,
        )
        return task_id

    queued = await asyncio.to_thread(_enqueue)
    return JSONResponse(
        status_code=202,
        content={
            "message": "Scheduled re-crawl accepted",
            "task_id": queued,
            "cadence": "scheduled re-crawl",
        },
    )
