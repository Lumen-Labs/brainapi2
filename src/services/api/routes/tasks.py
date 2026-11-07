"""
File: /tasks.py
Created Date: Saturday December 13th 2025
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Saturday December 13th 2025
Modified By: the developer formerly known as Christian Nonis at <alch.infoemail@gmail.com>
-----
"""

import json
from typing import Literal

from pydantic import BaseModel
from src.core.saving import ingest_cancellation
from src.utils.logging import log
from fastapi import APIRouter, Depends, HTTPException, Request, Query
from src.services.api.dependencies import get_brain_id
from src.services.kg_agent.main import cache_adapter
from src.services.api.constants.responses import TaskListResponse, TaskStateResponse, IngestionActivityResponse

tasks_router = APIRouter(prefix="/tasks", tags=["tasks"])


@tasks_router.get("/", response_model=TaskListResponse)
async def get_tasks(brain_id: str = Depends(get_brain_id)):
    try:
        task_keys = cache_adapter.get_task_keys(brain_id)
        results = []
        for task_key in task_keys:
            task_id = task_key.split(":")[-1]
            str_result = cache_adapter.get_task(task_id, brain_id=brain_id)
            if str_result is None:
                continue
            result = json.loads(str_result)
            results.append(
                {
                    **ingest_cancellation.public_state(
                        brain_id, task_id,
                        {**customer_task(result, brain_id), "status": result.get("status", "unknown")},
                    ),
                    "id": task_id,
                }
            )
        return {"tasks": results}
    except Exception as e:
        log(f"Error in get_tasks: {type(e).__name__}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Could not retrieve tasks",
        ) from e


@tasks_router.get("/{task_id}", response_model=TaskStateResponse)
async def get_task(task_id: str, brain_id: str = Depends(get_brain_id)):
    """
    Get the result of a task by its ID.
    """
    try:
        str_result = cache_adapter.get_task(task_id, brain_id=brain_id)
        if str_result is None:
            durable = ingest_cancellation.public_state(brain_id, task_id, {})
            if not durable:
                raise HTTPException(status_code=404, detail="Task not found")
            return {**durable, "task_id": task_id}
        if isinstance(str_result, bytes):
            result = json.loads(str_result.decode("utf-8"))
        else:
            result = json.loads(str_result)
        return {
            **ingest_cancellation.public_state(
                brain_id, task_id,
                {**customer_task(result, brain_id), "status": result.get("status", "unknown")},
            ),
            "task_id": task_id,
        }
    except HTTPException:
        raise
    except Exception as e:
        log(f"Error in get_task: {type(e).__name__}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Could not retrieve task",
        ) from e


class CancelIngestionRequest(BaseModel):
    mode: Literal["keep_saved", "rollback"]


class CancelIngestionResponse(BaseModel):
    task_id: str
    mode: Literal["keep_saved", "rollback"]
    status: Literal["cancelling", "cancelled", "rollback_failed"]


@tasks_router.post("/{task_id}/cancel", status_code=202, response_model=CancelIngestionResponse)
async def cancel_ingestion(
    task_id: str,
    body: CancelIngestionRequest,
    brain_id: str = Depends(get_brain_id),
):
    if not ingest_cancellation.task_control_enabled():
        raise HTTPException(404, "Cancellable ingestion not found")
    try:
        run = ingest_cancellation.request_cancellation(brain_id, task_id, body.mode)
    except ingest_cancellation.CancellationConflict as exc:
        raise HTTPException(409, "Cancellation mode is unavailable for this ingestion") from exc
    if run is None:
        raise HTTPException(404, "Cancellable ingestion not found")
    return {"task_id": task_id, "mode": run["mode"], "status": ingest_cancellation.cancellation_status(run)}


@tasks_router.get("/{task_id}/activity", response_model=IngestionActivityResponse)
async def get_activity(task_id: str, offset: int = Query(0, ge=0), limit: int = Query(50, ge=1, le=100), brain_id: str = Depends(get_brain_id)):
    from src.core.saving.ingest_manifest import manifest_snapshot
    result = manifest_snapshot(brain_id, task_id, offset, limit)
    if result is None:
        raise HTTPException(404, "Activity is not ready")
    return result


@tasks_router.post("/{task_id}/control", include_in_schema=False)
async def control_import(task_id: str, request: Request, brain_id: str = Depends(get_brain_id)):
    from src.core.saving.ingest_manifest import verify_control, manifest_snapshot, dispatch_sections, retry_sections
    body = await request.json()
    action = body.get("action", "snapshot")
    if not verify_control(brain_id, task_id, action, body.get("token")):
        raise HTTPException(403, "Platform control required")
    ids = body.get("section_ids") or []
    if not isinstance(ids, list) or len(ids) > 100 or any(not isinstance(v, str) for v in ids):
        raise HTTPException(422, "Invalid section ids")
    if action == "dispatch":
        dispatch_sections(brain_id, task_id, ids, body.get("routing"))
    elif action == "retry":
        retry_sections(brain_id, task_id, ids)
    elif action != "snapshot":
        raise HTTPException(422, "Unknown control action")
    result = manifest_snapshot(brain_id, task_id, max(0, int(body.get("offset", 0))), 100, internal=True)
    if result is None:
        raise HTTPException(404, "Activity is not ready")

    return result


def customer_task(result, brain_id=None):
    if brain_id and (result.get("counts") or {}).get("manifest_version"):
        from src.core.saving.ingest_manifest import manifest_snapshot
        snapshot = manifest_snapshot(brain_id,result.get("task_id"),limit=1)
        if snapshot:
            result = {**result,"status":snapshot["status"],"stage":snapshot["stage"],"progress":snapshot["progress"]}
    # Task responses are also accessible through normal BrainPATs.
    def scrub(value):
        if isinstance(value,dict):
            return {key:scrub(item) for key,item in value.items() if not any(term in key.lower() for term in ("token", "cost", "configuration", "routing")) and key not in {"usage", "usage_events", "provider", "model"}}
        if isinstance(value,list):
            return [scrub(item) for item in value]
        return value
    return scrub(result)


@tasks_router.post("/{task_id}/usage", include_in_schema=False)
async def task_usage(task_id: str, request: Request, brain_id: str = Depends(get_brain_id)):
    from src.core.saving.ingest_manifest import verify_control
    from src.services.data.main import data_adapter
    body = await request.json()
    if not verify_control(brain_id, task_id, "usage", body.get("token")):
        raise HTTPException(403, "Platform administration required")
    offset = max(0, int(body.get("offset", 0)))
    from src.core.saving.pipeline_store import collection
    events = collection(brain_id, "pipeline_usage_events")
    ack_ids = body.get("ack_ids")
    if ack_ids is not None:
        if not isinstance(ack_ids,list) or len(ack_ids)>100 or any(not isinstance(v,str) for v in ack_ids):
            raise HTTPException(422,"Invalid acknowledgement")
        events.update_many({"operation_id":task_id,"_id":{"$in":ack_ids}},{"$set":{"exported":True}})
    rows = list(events.find({"operation_id":task_id,"exported":{"$ne":True}}).sort("_id",1).limit(100))
    return {"events":[{"id":str(row.pop("_id")), **row} for row in rows], "next_offset":0 if len(rows)==100 else None}
