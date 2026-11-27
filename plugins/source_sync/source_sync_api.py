from __future__ import annotations

import os
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from source_sync_apply import (
    apply_event,
    apply_poll,
    approve_pending_deletes,
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


class ApproveDeletesBody(BaseModel):
    pending_delete_count: int
    pending_delete_hash: str


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

    @router.post("/sources/{source_id}/approve-deletes")
    def approve_source_deletes(
        source_id: str,
        body: ApproveDeletesBody,
        brain_id: str = Depends(_brain_id),
    ):
        store = _store()
        try:
            record = approve_pending_deletes(
                brain_id,
                source_id,
                count=body.pending_delete_count,
                digest=body.pending_delete_hash,
                store=store,
            )
        except ValueError as exc:
            detail = str(exc)
            status = 404 if detail == "source not found" else 409
            raise HTTPException(status_code=status, detail=detail) from exc
        task_id = _enqueue_poll(source_id)
        return {
            "status": "queued",
            "task_id": task_id,
            "source_id": source_id,
            "approved": True,
            "pending_delete_count": int(record.get("pending_delete_count") or 0),
            "pending_delete_hash": record.get("pending_delete_hash"),
        }

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
    brain_id = str(payload.get("brain_id") or "")
    record = store.get(brain_id, source_id) if brain_id else store.get_any(source_id)
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


def lookup_source_for_brain(store, brain_id: str, source_id: str) -> dict[str, Any]:
    record = store.get(brain_id, source_id)
    if record is None:
        raise ValueError("source not found")
    return record


def _mcp_brain(brain_id: str) -> str:
    from src.core.intent.brain import resolve_brain
    from src.core.instances import data_adapter
    from src.services.mcp.main import auth_token_var
    from src.services.mcp.utils import guard_brainpat

    return resolve_brain(
        auth_token_var.get(),
        brain_id,
        data_adapter=data_adapter,
        guard=guard_brainpat,
    )


def source_sync_get_source(source_id: str, brain_id: str = "") -> dict[str, Any]:
    try:
        resolved = _mcp_brain(brain_id)
        record = lookup_source_for_brain(_store(), resolved, source_id)
    except Exception as exc:
        return _mcp_lookup_error(exc)
    return {"ok": True, "source": _public_with_webhook(record)}


def source_sync_approve_deletes(
    source_id: str,
    pending_delete_count: int,
    pending_delete_hash: str,
    brain_id: str = "",
) -> dict[str, Any]:
    try:
        resolved = _mcp_brain(brain_id)
        store = _store()
        lookup_source_for_brain(store, resolved, source_id)
        updated = approve_pending_deletes(
            resolved,
            source_id,
            count=pending_delete_count,
            digest=pending_delete_hash,
            store=store,
        )
    except Exception as exc:
        return _mcp_lookup_error(exc)
    task_id = _enqueue_poll(str(updated.get("id") or source_id))
    return {
        "ok": True,
        "status": "queued",
        "task_id": task_id,
        "source_id": updated.get("id"),
        "pending_delete_count": int(updated.get("pending_delete_count") or 0),
        "pending_delete_hash": updated.get("pending_delete_hash"),
    }


def _mcp_lookup_error(exc: BaseException) -> dict[str, Any]:
    detail = str(exc)
    if detail in {"no pending deletions", "pending deletions changed", "source lock busy"}:
        return {"ok": False, "error": detail}
    return {"ok": False, "error": "source not found"}


def register_mcp_tools(context) -> None:
    register = getattr(context, "register_mcp_tool", None)
    if not callable(register):
        return

    register(
        source_sync_get_source,
        name="source_sync_get_source",
        description="Get a source-sync source and its pending-delete or truncation status.",
    )
    register(
        source_sync_approve_deletes,
        name="source_sync_approve_deletes",
        description=(
            "One-shot approval to apply pending source-sync deletions that exceeded "
            "the delete cap, then queue a poll. Pass pending_delete_count and "
            "pending_delete_hash from source_sync_get_source."
        ),
    )
