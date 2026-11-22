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
import io
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
from src.services.api.routes import spreadsheets
from src.core.saving.ingest_plain_text import (
    decode_plain_text,
    uses_plain_text_ingest,
)
from src.constants.tasks.ingestion import IngestionTaskArgs
from src.core.saving.ingest_staging import (
    REMOTE_OCR_EMPTY_ERROR,
    REMOTE_OCR_REJECTED_ERROR,
    fail_expired_remote_ocr,
    remote_ocr_callback_deadline,
)

MAX_TASK_RETRIES = 3
RETRY_DELAY_BASE = 0.1
DEFAULT_STAGED_UPLOAD_MAX_BYTES = 12 * 1024 * 1024
logger = logging.getLogger(__name__)


def staged_upload_max_bytes():
    raw = os.getenv("STAGED_UPLOAD_MAX_BYTES")
    if raw is None or not str(raw).strip():
        return DEFAULT_STAGED_UPLOAD_MAX_BYTES
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError):
        logger.warning(
            "Invalid STAGED_UPLOAD_MAX_BYTES %r; using %s",
            raw, DEFAULT_STAGED_UPLOAD_MAX_BYTES,
        )
        return DEFAULT_STAGED_UPLOAD_MAX_BYTES
    if value <= 0:
        logger.warning(
            "Non-positive STAGED_UPLOAD_MAX_BYTES %r; using %s",
            raw, DEFAULT_STAGED_UPLOAD_MAX_BYTES,
        )
        return DEFAULT_STAGED_UPLOAD_MAX_BYTES
    return value


def _read_upload_capped(upload, limit):
    chunks = []
    total = 0
    while True:
        chunk = upload.file.read(1024 * 1024)
        if not chunk:
            break
        total += len(chunk)
        if total > limit:
            raise HTTPException(413, "Staged file upload is too large")
        chunks.append(chunk)
    return b"".join(chunks)


def _upload_size(upload, content):
    if content is not None:
        return len(content)
    size = getattr(upload, "size", None)
    if isinstance(size, int) and size >= 0:
        return size
    return None


def _plain_text_payload(brain_id, content, group_key):
    payload = {
        "data": {"data_type": "text", "text_data": decode_plain_text(content)},
        "brain_id": brain_id,
    }
    if group_key:
        payload["group_tracking"] = group_key
    return payload


def _file_accepted(task_id):
    return JSONResponse(
        status_code=202,
        content={"message": "File ingestion accepted", "task_id": task_id},
    )


async def _queue_docling_file(brain_id, task, filename, content, context, group_key):
    content_b64 = base64.b64encode(content).decode("ascii")
    set_ingestion_task_status(
        task, brain_id, "queued", stage="queued", group_tracking=group_key,
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
        return
    for attempt in range(MAX_TASK_RETRIES):
        try:
            ingest_file_task.apply_async(
                args=[content_b64, filename, brain_id, group_key],
                task_id=task,
                **({"headers":{"pipeline_operation":context}} if context else {}),
            )
            return
        except OperationalError:
            if attempt == MAX_TASK_RETRIES - 1:
                raise HTTPException(status_code=503, detail="Task queue unavailable")
            await asyncio.sleep(RETRY_DELAY_BASE * (attempt + 1))


def _forward_remote_ocr(request, brain_id, task, file, filename, content, group_key):
    app_host = "{}://{}".format(
        request.headers.get("X-Forwarded-Proto") or request.url.scheme,
        request.headers.get("X-Forwarded-Host")
        or request.headers.get("Host")
        or request.base_url.netloc,
    ).rstrip("/")
    print(f"app_host: {app_host}")
    deadline = remote_ocr_callback_deadline(_upload_size(file, content))
    set_ingestion_task_status(
        task,
        brain_id,
        "queued",
        stage="queued",
        group_tracking=group_key,
        extra={"remote_ocr_deadline": deadline},
    )
    if (
        os.getenv("STAGED_INGESTION_ENABLED") == "1"
        and os.getenv("STAGED_ENRICHMENT_ENABLED") == "1"
    ):
        from src.core.saving import ingest_staging

        ingest_staging.create_run(brain_id, task, callback_deadline=deadline)
    parser_data = {
        "brain_id": brain_id,
        "webhook_callback": f"{app_host}/ingest/",
        "identifier": task,
    }
    if group_key:
        parser_data["group_tracking"] = group_key
    upload = io.BytesIO(content) if content is not None else file.file
    response = requests.post(
        f"{config.docparser_endpoint}/ingest",
        files={"file": (filename, upload, file.content_type)},
        data=parser_data,
        headers={"Authorization": f"Bearer {config.docparser_token}"},
    )
    if response.status_code != 200:
        fail_expired_remote_ocr(brain_id, task, rejected=True)
        raise HTTPException(status_code=503, detail=REMOTE_OCR_REJECTED_ERROR)


async def _queue_text_ingest(brain_id, task_id, payload, context, group_tracking=None):
    set_ingestion_task_status(
        task_id,
        brain_id,
        "queued",
        stage="queued",
        group_tracking=group_tracking,
    )
    if os.getenv("STAGED_INGESTION_ENABLED") == "1" and (
        payload.get("skip_enrichment") or os.getenv("STAGED_ENRICHMENT_ENABLED") == "1"
    ):
        _submit_staged(brain_id, task_id, payload, context)
        return
    headers = register(brain_id, task_id)
    if context:
        headers["pipeline_operation"] = context
    for attempt in range(MAX_TASK_RETRIES):
        try:
            ingest_data_task.apply_async(
                args=[payload],
                task_id=task_id,
                **({"headers": headers} if headers else {}),
            )
            return
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


def _ocr_callback_text(data):
    payload = getattr(data, "data", None)
    text = getattr(payload, "text_data", None) if payload is not None else None
    if isinstance(text, str):
        return text.strip()
    return ""


def _ocr_callback_failed(data):
    if str(getattr(data, "status", "") or "").lower() == "failed":
        return True
    return not _ocr_callback_text(data)


def _ocr_page_meta(data):
    return getattr(data, "page_index", None), getattr(data, "total_pages", None)


def _read_task_cache(brain_id, task_id):
    raw = cache_adapter.get(f"task:{task_id}", brain_id=brain_id)
    if not raw:
        getter = getattr(cache_adapter, "get_task", None)
        if callable(getter):
            raw = getter(task_id, brain_id=brain_id)
    if not raw:
        return None
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except Exception:
            return None
    return dict(raw)


_OCR_CALLBACK_FIELDS = {"status", "error", "page_index", "total_pages"}


def _ocr_callback_accepted(task_id):
    return JSONResponse(
        status_code=202,
        content={"message": "Ingestion accepted", "task_id": task_id},
    )


def _ingest_task_payload(data):
    return IngestionTaskArgs.model_validate(
        data.model_dump(exclude=_OCR_CALLBACK_FIELDS)
    ).model_dump(mode="json")


def _accept_remote_ocr_callback(brain_id, task_id, data, context):
    from src.core.saving import ingest_staging

    fail_expired_remote_ocr(brain_id, task_id)
    page_index, total_pages = _ocr_page_meta(data)
    if _ocr_callback_failed(data):
        error = (getattr(data, "error", None) or "").strip() or REMOTE_OCR_EMPTY_ERROR
        ingest_staging.record_ocr_page_failure(
            brain_id, task_id, error, page_index=page_index, total_pages=total_pages,
        )
        return _ocr_callback_accepted(task_id)
    _submit_staged(
        brain_id, task_id, _ingest_task_payload(data), context, existing_ocr=True,
        page_index=page_index, total_pages=total_pages,
    )
    return _ocr_callback_accepted(task_id)


def _accept_known_failed_callback(brain_id, task_id):
    if _read_task_cache(brain_id, task_id):
        return _ocr_callback_accepted(task_id)
    raise HTTPException(status_code=422, detail="data is required")


async def _try_spreadsheet_import(file, brain_id, task, group_key):
    try:
        from src.services.api.routes.spreadsheets import try_import_file
    except ImportError:
        return None
    workbook = await try_import_file(file, brain_id, workbook_id=task)
    if workbook is None:
        return None
    try:
        set_ingestion_task_status(
            task, brain_id, "completed", stage="completed",
            counts={"sheets": len(workbook.get("sheets") or ())},
            group_tracking=group_key,
        )
    except Exception:
        logger.exception("Workbook import status write failed for task %s", task)
    return JSONResponse(
        status_code=202,
        content={"message": "Workbook imported", "task_id": task},
    )


ingest_router = APIRouter(prefix="/ingest", tags=["ingest"])


def _staged_ocr_unit(staging, brain_id, task_id, message, page_index, total_pages):
    registered = staging.register_ocr_callback(
        brain_id, task_id, message,
        page_index=page_index, total_pages=total_pages,
    )
    if registered is None:
        raise HTTPException(409, "This OCR import no longer accepts data")
    unit_id, created = registered
    return unit_id if created else None


def _staged_file_unit(staging, brain_id, task_id, message, payload, file_content):
    from src.core.saving.ingest_staged_input import save_file

    staging.create_run(brain_id, task_id, upload_deadline=time() + 3600)
    parts = save_file(brain_id, task_id, file_content)
    message["payload"] = {**payload, "content_b64": "", "content_parts": parts}
    unit_id = staging.register_unit(brain_id, task_id, "discovery", message)
    if unit_id is None:
        raise HTTPException(503, "Staged file input could not be accepted")
    return unit_id


def _submit_staged(
    brain_id, task_id, payload, context, *, kind="source", discovery=False,
    file_content=None, existing_ocr=False, page_index=None, total_pages=None,
):
    from src.core.saving import ingest_staging
    from src.core.saving.ingest_worker import dispatch_staged_source

    message = {"kind": kind, "payload": payload}
    if context:
        message["headers"] = {"pipeline_operation": context}
    if existing_ocr:
        unit_id = _staged_ocr_unit(
            ingest_staging, brain_id, task_id, message, page_index, total_pages,
        )
        if unit_id is None:
            return
    elif file_content is None:
        unit_id = ingest_staging.create_run(
            brain_id, task_id, message=message, discovery=discovery,
        )
    else:
        unit_id = _staged_file_unit(
            ingest_staging, brain_id, task_id, message, payload, file_content,
        )
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
        from src.core.saving import ingest_staging

        if ingest_staging.is_remote_ocr_run(brain_id, task_id):
            return _accept_remote_ocr_callback(brain_id, task_id, data, context)

    if str(getattr(data, "status", "") or "").lower() == "failed" or data.data is None:
        if flow_task_identifier:
            return _accept_known_failed_callback(brain_id, task_id)
        raise HTTPException(status_code=422, detail="data is required")

    extra = {"ocr_pages_accepted": True} if flow_task_identifier else None
    set_ingestion_task_status(
        task_id,
        data.brain_id,
        "queued",
        stage="queued",
        group_tracking=data.group_tracking,
        extra=extra,
    )

    if os.getenv("STAGED_INGESTION_ENABLED") == "1" and (
        data.skip_enrichment or os.getenv("STAGED_ENRICHMENT_ENABLED") == "1"
    ):
        _submit_staged(brain_id, task_id, _ingest_task_payload(data), context)
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
                args=[_ingest_task_payload(data)],
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
    if len(await request.body()) > staged_upload_max_bytes():
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

    imported = await _try_spreadsheet_import(file, brain_id, task, group_key)
    if imported is not None:
        return imported
    try:
        await _dispatch_file_ingest(request, file, brain_id, task, context, group_key)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail="File ingestion failed") from exc
    return _file_accepted(task)


async def _dispatch_file_ingest(request, file, brain_id, task, context, group_key):
    filename = file.filename or "file"
    file.file.seek(0)
    staged_file = (
        os.getenv("STAGED_INGESTION_ENABLED") == "1"
        and os.getenv("STAGED_ENRICHMENT_ENABLED") == "1"
    )
    plain_text = uses_plain_text_ingest(brain_id, filename, file.content_type)
    content = None
    if staged_file:
        content = _read_upload_capped(file, staged_upload_max_bytes())
    elif plain_text:
        content = file.file.read()
    if content is not None:
        plain_text = uses_plain_text_ingest(
            brain_id, filename, file.content_type, content,
        )
    if plain_text:
        await _queue_text_ingest(
            brain_id,
            task,
            _plain_text_payload(brain_id, content or b"", group_key),
            context,
            group_key,
        )
        return
    if config.ocr_mode == "docling":
        payload = file.file.read() if content is None else content
        await _queue_docling_file(
            brain_id, task, filename, payload, context, group_key,
        )
        return
    _forward_remote_ocr(
        request, brain_id, task, file, filename, content, group_key,
    )
