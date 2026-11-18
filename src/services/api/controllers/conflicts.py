"""REST handlers for cross-source fact conflicts."""

from __future__ import annotations

import asyncio
from typing import Optional
from uuid import uuid4

from fastapi import HTTPException
from starlette.responses import JSONResponse, PlainTextResponse

from src.core.saving.conflict_report import build_report
from src.core.saving.conflicts import (
    dismiss_conflict,
    get_conflict,
    list_conflicts,
    resolve_conflict,
)
from src.core.saving.validity import InvalidAsOf, optional_instant
from src.services.data.main import data_adapter


def _dump(conflict) -> dict:
    dump = getattr(conflict, "model_dump", None)
    if callable(dump):
        return dump(mode="json")
    return conflict


async def get_conflict_by_id(id: str, brain_id: str = "default"):
    def _load():
        conflict = get_conflict(data_adapter, brain_id, id)
        if conflict is None:
            raise HTTPException(status_code=404, detail="Conflict not found")
        return conflict

    result = await asyncio.to_thread(_load)
    return JSONResponse(
        content={
            "message": "Conflict retrieved successfully",
            "conflict": _dump(result),
        }
    )


async def get_conflicts_list(
    limit: int = 10,
    skip: int = 0,
    status: Optional[str] = None,
    source: Optional[str] = None,
    subject: Optional[str] = None,
    since: Optional[str] = None,
    until: Optional[str] = None,
    brain_id: str = "default",
):
    try:
        since_dt = optional_instant(since, name="since")
        until_dt = optional_instant(until, name="until")
    except InvalidAsOf as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if status and status not in {"open", "resolved", "dismissed"}:
        raise HTTPException(status_code=400, detail="status must be open, resolved, or dismissed")

    def _load():
        return list_conflicts(
            data_adapter,
            brain_id,
            status=status,
            source=source,
            subject=subject,
            since=since_dt,
            until=until_dt,
            limit=limit,
            skip=skip,
        )

    rows, total = await asyncio.to_thread(_load)
    return JSONResponse(
        content={
            "message": "Conflicts list retrieved successfully",
            "conflicts": [_dump(row) for row in rows],
            "count": len(rows),
            "total": total,
        }
    )


async def resolve_conflict_by_id(
    id: str,
    brain_id: str = "default",
    winning_fact_id: Optional[str] = None,
    notes: Optional[str] = None,
    method: str = "manual",
):
    def _resolve():
        try:
            from src.services.kg_agent.main import graph_adapter

            return resolve_conflict(
                data_adapter,
                brain_id,
                id,
                winning_fact_id=winning_fact_id,
                notes=notes,
                method=method,
                graph=graph_adapter,
            )
        except KeyError:
            raise HTTPException(status_code=404, detail="Conflict not found")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    result = await asyncio.to_thread(_resolve)
    return JSONResponse(
        content={
            "message": "Conflict resolved",
            "conflict": _dump(result),
        }
    )


async def dismiss_conflict_by_id(
    id: str,
    brain_id: str = "default",
    notes: Optional[str] = None,
):
    def _dismiss():
        try:
            return dismiss_conflict(data_adapter, brain_id, id, notes=notes)
        except KeyError:
            raise HTTPException(status_code=404, detail="Conflict not found")

    result = await asyncio.to_thread(_dismiss)
    return JSONResponse(
        content={
            "message": "Conflict dismissed",
            "conflict": _dump(result),
        }
    )


async def get_conflict_report(
    brain_id: str = "default",
    status: Optional[str] = None,
    source: Optional[str] = None,
    subject: Optional[str] = None,
    since: Optional[str] = None,
    until: Optional[str] = None,
    fmt: str = "json",
    limit: int = 200,
    skip: int = 0,
):
    try:
        since_dt = optional_instant(since, name="since")
        until_dt = optional_instant(until, name="until")
    except InvalidAsOf as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if fmt not in {"json", "markdown"}:
        raise HTTPException(status_code=400, detail="format must be json or markdown")

    def _build():
        return build_report(
            data_adapter,
            brain_id,
            status=status,
            source=source,
            subject=subject,
            since=since_dt,
            until=until_dt,
            limit=limit,
            skip=skip,
            fmt=fmt,
        )

    payload = await asyncio.to_thread(_build)
    if fmt == "markdown":
        return PlainTextResponse(content=payload.get("markdown") or "", media_type="text/markdown")
    return JSONResponse(
        content={
            "message": "Conflict report generated",
            **payload,
        }
    )


async def start_conflict_scan(
    brain_id: str = "default",
    use_llm: bool = False,
):
    from src.workers.tasks.conflicts import scan_conflicts
    from src.workers.tasks.ingestion import set_ingestion_task_status

    task_id = str(uuid4())

    def _enqueue():
        set_ingestion_task_status(task_id, brain_id, "queued", stage="conflict_scan")
        scan_conflicts.apply_async(
            args=[{"brain_id": brain_id, "use_llm": use_llm}],
            task_id=task_id,
        )
        return task_id

    queued = await asyncio.to_thread(_enqueue)
    return JSONResponse(
        status_code=202,
        content={"message": "Conflict scan accepted", "task_id": queued},
    )
