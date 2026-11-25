from __future__ import annotations

import os
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from source_sync_apply import (
    apply_event,
    apply_poll,
    event_from_payload,
    event_task_payload,
    queue_ingest_task,
)
from source_sync_connector import WebhookRejected, get_connector
from source_sync_store import public_source, default_store


class CreateSourceBody(BaseModel):
    name: str
    source_type: str = "zendesk_guide"
    sync_mode: str = "webhook"
    poll_interval_seconds: int = 3600
    base_url: str | None = None
    locale: str | None = None
    webhook_secret_env: str | None = None
    api_token_env: str | None = None
    api_email: str | None = None
    skip_enrichment: bool = False


class RecheckBody(BaseModel):
    force: bool = Field(default=False)


def _store():
    return default_store()


def _brain_id(request: Request) -> str:
    return str(getattr(request.state, "brain_id", None) or "default")


def _enqueue_apply(payload: dict[str, Any]) -> str:
    from src.workers.app import ingestion_app

    result = ingestion_app.send_task(
        "source_sync.apply_event",
        args=[payload],
        queue="source_sync",
    )
    return str(getattr(result, "id", "") or "")


def _enqueue_poll(source_id: str) -> str:
    from src.workers.app import ingestion_app

    result = ingestion_app.send_task(
        "source_sync.poll_source",
        args=[{"source_id": source_id}],
        queue="source_sync",
    )
    return str(getattr(result, "id", "") or "")


def create_admin_router() -> APIRouter:
    router = APIRouter(prefix="/source-sync", tags=["source-sync"])

    @router.get("/sources")
    def list_sources(brain_id: str = Depends(_brain_id)):
        sources = [_public_with_webhook(item) for item in _store().list(brain_id)]
        return {"sources": sources, "total": len(sources)}

    @router.post("/sources", status_code=201)
    def create_source(body: CreateSourceBody, brain_id: str = Depends(_brain_id)):
        try:
            record = _store().create(brain_id, body.model_dump())
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return _public_with_webhook(record)

    @router.get("/sources/{source_id}")
    def get_source(source_id: str, brain_id: str = Depends(_brain_id)):
        record = _store().get(brain_id, source_id)
        if record is None:
            raise HTTPException(status_code=404, detail="source not found")
        return _public_with_webhook(record)

    @router.delete("/sources/{source_id}")
    def delete_source(source_id: str, brain_id: str = Depends(_brain_id)):
        if not _store().delete(brain_id, source_id):
            raise HTTPException(status_code=404, detail="source not found")
        return {"ok": True}

    @router.post("/sources/{source_id}/recheck")
    def recheck_source(
        source_id: str,
        body: RecheckBody | None = None,
        brain_id: str = Depends(_brain_id),
    ):
        record = _store().get(brain_id, source_id)
        if record is None:
            raise HTTPException(status_code=404, detail="source not found")
        task_id = _enqueue_poll(source_id)
        return {"status": "queued", "task_id": task_id, "source_id": source_id}

    @router.get("/dead-letters")
    def list_dead_letters(brain_id: str = Depends(_brain_id)):
        items = _store().list_dead_letters(brain_id)
        return {"dead_letters": items, "total": len(items)}

    return router


def create_webhook_router() -> APIRouter:
    router = APIRouter(prefix="/source-sync/webhooks", tags=["source-sync"])

    @router.post("/{source_id}")
    async def receive_webhook(source_id: str, request: Request):
        store = _store()
        record = store.get_any(source_id)
        if record is None:
            raise HTTPException(status_code=404, detail="source not found")
        body = await request.body()
        headers = dict(request.headers.items())
        try:
            connector = get_connector(str(record.get("source_type") or ""))
            secret = _webhook_secret(record)
            if not connector.verify_webhook(headers, body, secret):
                raise HTTPException(status_code=401, detail="invalid webhook signature")
            events = connector.parse_webhook(headers, body)
        except WebhookRejected as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        queued = []
        for event in events:
            payload = event_task_payload(record, event)
            queued.append(
                {
                    "document_id": event.document_id,
                    "revision": event.revision,
                    "action": event.action,
                    "task_id": _enqueue_apply(payload),
                }
            )
        return {"status": "accepted", "count": len(queued), "events": queued}

    return router


def apply_payload_now(
    payload: dict[str, Any],
    *,
    store=None,
    data=None,
    graph=None,
    queue_ingest=queue_ingest_task,
) -> dict[str, Any]:
    store = store or _store()
    source_id = str(payload.get("source_id") or "")
    record = store.get_any(source_id)
    if record is None:
        raise ValueError("source not found")
    event = event_from_payload(payload)
    return apply_event(
        event,
        record,
        store=store,
        data=data,
        graph=graph,
        queue_ingest=queue_ingest,
    )


def poll_source_now(
    source_id: str,
    *,
    store=None,
    enqueue_event=_enqueue_apply,
) -> dict[str, Any]:
    store = store or _store()
    record = store.get_any(source_id)
    if record is None:
        raise ValueError("source not found")
    return apply_poll(record, store=store, enqueue_event=enqueue_event)


def _webhook_secret(record: dict[str, Any]) -> str:
    env_name = str(record.get("webhook_secret_env") or "").strip()
    if not env_name:
        return ""
    return os.getenv(env_name, "")


def _public_with_webhook(record: dict[str, Any]) -> dict[str, Any]:
    public = public_source(record)
    public["webhook_path"] = f"/source-sync/webhooks/{record.get('id')}"
    return public
