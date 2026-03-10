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
import base64
import json
from uuid import uuid4

import requests
from celery.exceptions import OperationalError
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from starlette.responses import JSONResponse
from typing_extensions import Annotated

from src.config import config
from src.services.api.constants.requests import (
    IngestionRequestBody,
    IngestionStructuredRequestBody,
)
from src.services.input.agents import cache_adapter
from src.workers.tasks.ingestion import ingest_data as ingest_data_task
from src.workers.tasks.ingestion import ingest_file as ingest_file_task
from src.workers.tasks.ingestion import (
    ingest_structured_data as ingest_structured_data_task,
)

MAX_TASK_RETRIES = 3
RETRY_DELAY_BASE = 0.1

ingest_router = APIRouter(prefix="/ingest", tags=["ingest"])


@ingest_router.post(path="/")
async def ingest_data(data: IngestionRequestBody, request: Request):
    """
    Ingest data to the processing pipeline and save to the memory.
    """

    flow_task_identifier = None

    if request.headers.get("Task-Identifier"):
        flow_task_identifier = request.headers.get("Task-Identifier")

    task = None
    for attempt in range(MAX_TASK_RETRIES):
        try:
            kwargs = {"args": [data.model_dump()]}
            if flow_task_identifier:
                kwargs["task_id"] = flow_task_identifier

            task = ingest_data_task.apply_async(**kwargs)
            break
        except OperationalError:
            if attempt == MAX_TASK_RETRIES - 1:
                raise HTTPException(status_code=503, detail="Task queue unavailable")
            await asyncio.sleep(RETRY_DELAY_BASE * (attempt + 1))

    cache_adapter.set(
        key=f"task:{task.id}",
        value=json.dumps({"status": "queued", "task_id": task.id}),
        brain_id=data.brain_id,
        expires_in=3600 * 24 * 7,
    )

    return JSONResponse(
        content={"message": "Data ingested successfully", "task_id": task.id}
    )


@ingest_router.post(path="/structured")
async def ingest_structured_data(data: IngestionStructuredRequestBody):
    """
    Ingest structured data to the processing pipeline and save to the memory.
    """
    task = None
    for attempt in range(MAX_TASK_RETRIES):
        try:
            task = ingest_structured_data_task.delay(data.model_dump())
            break
        except OperationalError:
            if attempt == MAX_TASK_RETRIES - 1:
                raise HTTPException(status_code=503, detail="Task queue unavailable")
            await asyncio.sleep(RETRY_DELAY_BASE * (attempt + 1))

    cache_adapter.set(
        key=f"task:{task.id}",
        value=json.dumps({"status": "queued", "task_id": task.id}),
        brain_id=data.brain_id,
        expires_in=3600 * 24 * 7,
    )

    return JSONResponse(
        content={"message": "Structured data ingested successfully", "task_id": task.id}
    )


@ingest_router.post(path="/file")
async def ingest_file(
    request: Request,
    file: Annotated[UploadFile, File()],
    brain_id: str = Form(default="default"),
):
    """
    Ingest a file into the processing pipeline and save to the memory.
    """

    task = str(uuid4())

    try:
        file.file.seek(0)

        if config.ocr_mode == "docling":
            content_b64 = base64.b64encode(file.file.read()).decode("ascii")
            filename = file.filename or "file"
            for attempt in range(MAX_TASK_RETRIES):
                try:
                    ingest_file_task.apply_async(
                        args=[content_b64, filename, brain_id],
                        task_id=task,
                    )
                    break
                except OperationalError:
                    if attempt == MAX_TASK_RETRIES - 1:
                        raise HTTPException(
                            status_code=503, detail="Task queue unavailable"
                        )
                    await asyncio.sleep(RETRY_DELAY_BASE * (attempt + 1))
            cache_adapter.set(
                key=f"task:{task}",
                value=json.dumps({"status": "queued", "task_id": task}),
                brain_id=brain_id,
                expires_in=3600 * 24 * 7,
            )
            response_content = {
                "message": "File ingested successfully",
                "task_id": task,
            }
        else:
            app_host = str(request.base_url).rstrip("/")
            cache_adapter.set(
                key=f"task:{task}",
                value=json.dumps({"status": "queued", "task_id": task}),
                brain_id=brain_id,
                expires_in=3600 * 24 * 7,
            )
            response = requests.post(
                f"{config.docparser_endpoint}/ingest",
                files={"file": (file.filename or "file", file.file, file.content_type)},
                data={
                    "brain_id": brain_id,
                    "webhook_callback": f"{app_host}/ingest",
                    "identifier": task,
                },
                headers={"Authorization": f"Bearer {config.docparser_token}"},
            )
            if response.status_code != 200:
                raise HTTPException(status_code=500, detail=response.text)
            response_content = {
                "message": "File ingested successfully",
                "task_id": task,
            }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return JSONResponse(content=response_content)
