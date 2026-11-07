"""
File: /ingest.py
Created Date: Sunday October 19th 2025
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Wednesday March 4th 2026 9:35:41 pm
Modified By: Christian Nonis <alch.infoemail@gmail.com>
-----
"""

import asyncio
import logging
import os
from time import time
import base64
import json
from uuid import uuid4

import requests
from celery.exceptions import OperationalError
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from starlette.responses import JSONResponse
from typing_extensions import Annotated

from src.config import config
from src.services.api.dependencies import get_brain_id
from src.services.api.constants.requests import (
    IngestionRequestBody,
    IngestionStructuredRequestBody,
    IngestionUrlRequestBody,
)
from src.core.saving.ingest_groups import InvalidGroupTracking, normalize_group_tracking
from src.services.api.constants.responses import IngestionAcceptedResponse
from src.services.input.agents import cache_adapter
from src.core.saving.ingest_cancellation import register
from src.workers.tasks.ingestion import ingest_data as ingest_data_task
from src.workers.tasks.ingestion import ingest_file as ingest_file_task
from src.workers.tasks.ingestion import ingest_url as ingest_url_task
from src.workers.tasks.ingestion import (
    ingest_structured_data as ingest_structured_data_task,
)
from src.workers.tasks.ingestion import set_ingestion_task_status

MAX_TASK_RETRIES = 3
RETRY_DELAY_BASE = 0.1

ingest_router = APIRouter(prefix="/ingest", tags=["ingest"])
logger = logging.getLogger(__name__)


def _submit_staged(
    brain_id, task_id, payload, context, *, kind="source", discovery=False,
    file_content=None, existing_ocr=False,
):
    from src.core.saving import ingest_staging
    from src.core.saving.ingest_worker import dispatch_staged_source

    message = {"kind": kind, "payload": payload}
    if context:
        message["headers"] = {"pipeline_operation": context}
    if existing_ocr:
        unit_id = ingest_staging.register_unit(brain_id, task_id, "source", message, seal=True)
        if unit_id is None:
            raise HTTPException(409, "This OCR import no longer accepts data")
    elif file_content is None:
        unit_id = ingest_staging.create_run(
            brain_id, task_id, message=message, discovery=discovery,
        )
    else:
        from src.core.saving.ingest_staged_input import save_file

        ingest_staging.create_run(brain_id, task_id, upload_deadline=time() + 3600)
        parts = save_file(brain_id, task_id, file_content)
        message["payload"] = {**payload, "content_b64": "", "content_parts": parts}
        unit_id = ingest_staging.register_unit(brain_id, task_id, "discovery", message)
        if unit_id is None:
            raise HTTPException(503, "Staged file input could not be accepted")
    try:
        dispatch_staged_source(brain_id, task_id, unit_id)
    except OperationalError:
        logger.warning("Staged ingestion %s will be dispatched by reconciliation", task_id)


@ingest_router.post(
    path="/", status_code=202, response_model=IngestionAcceptedResponse
)
async def ingest_data(
    data: IngestionRequestBody,
    request: Request,
    brain_id: str = Depends(get_brain_id),
):
    """
    Accept data for asynchronous ingestion.
    """
    data.brain_id = brain_id
    print("[Ingest] received task for brain: ", brain_id)

    flow_task_identifier = request.headers.get("Task-Identifier")
    task_id = flow_task_identifier or str(uuid4())
    from src.core.saving.ingest_manifest import pipeline_context
    context = pipeline_context(request, brain_id, task_id)

    if flow_task_identifier and os.getenv("STAGED_INGESTION_ENABLED") == "1":
        from src.core.saving.ingest_staging import is_remote_ocr_run

        if is_remote_ocr_run(brain_id, task_id):
            _submit_staged(
                brain_id, task_id, data.model_dump(mode="json"), context, existing_ocr=True,
            )
            return JSONResponse(
                status_code=202,
                content={"message": "Ingestion accepted", "task_id": task_id},
            )

    set_ingestion_task_status(
        task_id,
        data.brain_id,
        "queued",
        stage="queued",
        group_tracking=data.group_tracking,
    )

    if os.getenv("STAGED_INGESTION_ENABLED") == "1" and (
        data.skip_enrichment or os.getenv("STAGED_ENRICHMENT_ENABLED") == "1"
    ):
        _submit_staged(brain_id, task_id, data.model_dump(mode="json"), context)
        return JSONResponse(
            status_code=202,
            content={"message": "Ingestion accepted", "task_id": task_id},
        )

    headers = register(brain_id, task_id)
    if context:
        headers["pipeline_operation"] = context

    for attempt in range(MAX_TASK_RETRIES):
        try:
            ingest_data_task.apply_async(
                args=[data.model_dump()],
                task_id=task_id,
                **({"headers": headers} if headers else {}),
            )
            break
        except OperationalError:
            if attempt == MAX_TASK_RETRIES - 1:
                set_ingestion_task_status(
                    task_id,
                    data.brain_id,
                    "failed",
                    stage="queue",
                    error="Task queue unavailable",
                )
                raise HTTPException(status_code=503, detail="Task queue unavailable")
            await asyncio.sleep(RETRY_DELAY_BASE * (attempt + 1))

    return JSONResponse(
        status_code=202,
        content={"message": "Ingestion accepted", "task_id": task_id},
    )


@ingest_router.post(
    path="/structured", status_code=202, response_model=IngestionAcceptedResponse
)
async def ingest_structured_data(
    data: IngestionStructuredRequestBody,
    request: Request,
    brain_id: str = Depends(get_brain_id),
):
    """
    Accept structured data for asynchronous ingestion.
    """
    data.brain_id = brain_id
    print("[IngestStructured] received task for brain: ", brain_id)
    task_id = str(uuid4())
    from src.core.saving.ingest_manifest import pipeline_context
    context = pipeline_context(request, brain_id, task_id)
    set_ingestion_task_status(
        task_id,
        data.brain_id,
        "queued",
        stage="queued",
        group_tracking=data.group_tracking,
    )

    if os.getenv("STAGED_INGESTION_ENABLED") == "1" and os.getenv("STAGED_ENRICHMENT_ENABLED") == "1":
        _submit_staged(brain_id, task_id, data.model_dump(mode="json"), context, kind="structured")
        return JSONResponse(
            status_code=202,
            content={"message": "Structured ingestion accepted", "task_id": task_id},
        )

    headers = register(brain_id, task_id)
    if context:
        headers["pipeline_operation"] = context

    for attempt in range(MAX_TASK_RETRIES):
        try:
            ingest_structured_data_task.apply_async(
                args=[data.model_dump()],
                task_id=task_id,
                **({"headers": headers} if headers else {}),
            )
            break
        except OperationalError:
            if attempt == MAX_TASK_RETRIES - 1:
                set_ingestion_task_status(
                    task_id,
                    data.brain_id,
                    "failed",
                    stage="queue",
                    error="Task queue unavailable",
                )
                raise HTTPException(status_code=503, detail="Task queue unavailable")
            await asyncio.sleep(RETRY_DELAY_BASE * (attempt + 1))

    return JSONResponse(
        status_code=202,
        content={"message": "Structured ingestion accepted", "task_id": task_id},
    )


@ingest_router.post("/structured/batch", status_code=202, response_model=IngestionAcceptedResponse)
async def ingest_structured_batch(
    batches: list[IngestionStructuredRequestBody],
    request: Request,
    brain_id: str = Depends(get_brain_id),
):
    if os.getenv("STAGED_INGESTION_ENABLED") != "1" or os.getenv("STAGED_ENRICHMENT_ENABLED") != "1":
        raise HTTPException(404, "Staged structured batches are unavailable")
    if not batches or len(batches) > 100:
        raise HTTPException(422, "Provide between 1 and 100 structured batches")
    if len(await request.body()) > 12 * 1024 * 1024:
        raise HTTPException(413, "Staged structured batch is too large")
    from src.core.saving.ingest_staging import create_batch_run
    from src.core.saving.ingest_manifest import pipeline_context

    task_id = str(uuid4())
    context = pipeline_context(request, brain_id, task_id)
    messages = []
    for batch in batches:
        batch.brain_id = brain_id
        message = {"kind": "structured", "payload": batch.model_dump(mode="json")}
        if context:
            message["headers"] = {"pipeline_operation": context}
        messages.append(message)
    set_ingestion_task_status(task_id, brain_id, "queued", stage="queued")
    create_batch_run(brain_id, task_id, messages)
    return {"message": "Structured ingestion accepted", "task_id": task_id}


@ingest_router.post(
    path="/url", status_code=202, response_model=IngestionAcceptedResponse
)
async def ingest_url(
    data: IngestionUrlRequestBody,
    request: Request,
    brain_id: str = Depends(get_brain_id),
):
    """
    Accept a public https URL for asynchronous crawl-and-ingest.
    """
    if not config.crawler_is_configured:
        raise HTTPException(status_code=503, detail="URL crawler is not configured")

    if data.progressive:
        from src.core.saving.ingest_manifest import pipeline_context
        if not pipeline_context(request, brain_id, "accept") :
            raise HTTPException(403, "Platform settlement required")
        if os.getenv("PROGRESSIVE_INGESTION_ENABLED") != "1":
            raise HTTPException(409, "Progressive ingestion is not enabled")
    payload = data.model_dump()
    payload["brain_id"] = brain_id
    task_id = str(uuid4())
    from src.core.saving.ingest_manifest import pipeline_context
    context = pipeline_context(request, brain_id, task_id)
    if context:
        payload["pipeline_configuration"] = context["configuration"]
    set_ingestion_task_status(
        task_id,
        brain_id,
        "queued",
        stage="queued",
        group_tracking=data.group_tracking,
    )

    if os.getenv("STAGED_INGESTION_ENABLED") == "1" and (
        payload.get("skip_enrichment") or os.getenv("STAGED_ENRICHMENT_ENABLED") == "1"
    ):
        _submit_staged(brain_id, task_id, payload, context, kind="url_discovery", discovery=True)
        return JSONResponse(
            status_code=202,
            content={"message": "URL ingestion accepted", "task_id": task_id},
        )

    for attempt in range(MAX_TASK_RETRIES):
        try:
            ingest_url_task.apply_async(
                args=[payload],
                task_id=task_id,
                **({"headers":{"pipeline_operation":context}} if context else {}),
            )
            break
        except OperationalError:
            if attempt == MAX_TASK_RETRIES - 1:
                set_ingestion_task_status(
                    task_id,
                    brain_id,
                    "failed",
                    stage="queue",
                    error="Task queue unavailable",
                )
                raise HTTPException(status_code=503, detail="Task queue unavailable")
            await asyncio.sleep(RETRY_DELAY_BASE * (attempt + 1))

    return JSONResponse(
        status_code=202,
        content={"message": "URL ingestion accepted", "task_id": task_id},
    )


@ingest_router.post(
    path="/file", status_code=202, response_model=IngestionAcceptedResponse
)
async def ingest_file(
    request: Request,
    file: Annotated[UploadFile, File()],
    brain_id: str = Depends(get_brain_id),
    group_tracking: Annotated[str | None, Form()] = None,
):
    """
    Ingest a file into the processing pipeline and save to the memory.
    """
    print("[IngestFile] received task for brain: ", brain_id)

    task = str(uuid4())
    from src.core.saving.ingest_manifest import pipeline_context
    context = pipeline_context(request, brain_id, task)
    raw_group = group_tracking or request.query_params.get("group_tracking")
    try:
        group_key = normalize_group_tracking(raw_group, strict=True)
    except InvalidGroupTracking as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    try:
        file.file.seek(0)

        if config.ocr_mode == "docling":
            content_b64 = base64.b64encode(file.file.read()).decode("ascii")
            filename = file.filename or "file"
            set_ingestion_task_status(
                task,
                brain_id,
                "queued",
                stage="queued",
                group_tracking=group_key,
            )
            if (
                os.getenv("STAGED_INGESTION_ENABLED") == "1"
                and os.getenv("STAGED_ENRICHMENT_ENABLED") == "1"
            ):
                _submit_staged(
                    brain_id, task,
                    {"filename": filename, "group_tracking": group_key},
                    context, kind="file_discovery", discovery=True,
                    file_content=content_b64,
                )
                return JSONResponse(
                    status_code=202,
                    content={"message": "File ingestion accepted", "task_id": task},
                )
            for attempt in range(MAX_TASK_RETRIES):
                try:
                    ingest_file_task.apply_async(
                        args=[content_b64, filename, brain_id, group_key],
                        task_id=task,
                        **({"headers":{"pipeline_operation":context}} if context else {}),
                    )
                    break
                except OperationalError:
                    if attempt == MAX_TASK_RETRIES - 1:
                        raise HTTPException(
                            status_code=503, detail="Task queue unavailable"
                        )
                    await asyncio.sleep(RETRY_DELAY_BASE * (attempt + 1))
            response_content = {
                "message": "File ingestion accepted",
                "task_id": task,
            }
        else:
            forwarded_proto = (
                request.headers.get("X-Forwarded-Proto") or request.url.scheme
            )
            forwarded_host = (
                request.headers.get("X-Forwarded-Host")
                or request.headers.get("Host")
                or request.base_url.netloc
            )
            app_host = f"{forwarded_proto}://{forwarded_host}".rstrip("/")
            print(f"app_host: {app_host}")
            set_ingestion_task_status(
                task,
                brain_id,
                "queued",
                stage="queued",
                group_tracking=group_key,
            )
            staged_ocr = (
                os.getenv("STAGED_INGESTION_ENABLED") == "1"
                and os.getenv("STAGED_ENRICHMENT_ENABLED") == "1"
            )
            if staged_ocr:
                from src.core.saving import ingest_staging

                ingest_staging.create_run(brain_id, task, callback_deadline=time() + 3600)
            parser_data = {
                "brain_id": brain_id,
                "webhook_callback": f"{app_host}/ingest/",
                "identifier": task,
            }
            if group_key:
                parser_data["group_tracking"] = group_key
            response = requests.post(
                f"{config.docparser_endpoint}/ingest",
                files={"file": (file.filename or "file", file.file, file.content_type)},
                data=parser_data,
                headers={"Authorization": f"Bearer {config.docparser_token}"},
            )
            if response.status_code != 200:
                if staged_ocr:
                    ingest_staging.expire_remote_ocr(brain_id, task, rejected=True)
                raise HTTPException(
                    status_code=503,
                    detail="Document parser rejected the ingestion request",
                )
            response_content = {
                "message": "File ingestion accepted",
                "task_id": task,
            }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail="File ingestion failed") from exc

    return JSONResponse(status_code=202, content=response_content)
