"""
File: /ingest.py
Created Date: Sunday October 19th 2025
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Monday January 5th 2026 7:57:27 pm
Modified By: Christian Nonis <alch.infoemail@gmail.com>
-----
"""

import json
import asyncio
import requests
from uuid import uuid4
from fastapi import APIRouter, HTTPException, Request
from typing import Literal
from typing_extensions import Annotated
from fastapi import File, Form, UploadFile
from starlette.responses import JSONResponse
from celery.exceptions import OperationalError

from src.config import config
from src.services.api.constants.requests import (
    IngestionRequestBody,
    IngestionStructuredRequestBody,
)
from src.services.input.agents import cache_adapter
from src.workers.tasks.ingestion import ingest_data as ingest_data_task
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
    file: Annotated[UploadFile, File()],
    brain_id: str = Form(default="default"),
):
    """
    Ingest a file into the processing pipeline and save to the memory.
    """

    task = str(uuid4())

    cache_adapter.set(
        key=f"task:{task}",
        value=json.dumps({"status": "queued", "task_id": task}),
        brain_id=brain_id,
        expires_in=3600 * 24 * 7,
    )

    try:
        file.file.seek(0)

        # We are forwarding to the DocParser API to convert files into markdown text
        # and by forwarding the webhook (our app host) and the task identifier,
        # we can be called back from the DocParser API to start the ingestion process
        # with the created markdown text.
        response = requests.post(
            f"{config.docparser_endpoint}/ingest",
            files={"file": (file.filename or "file", file.file, file.content_type)},
            data={
                "brain_id": brain_id,
                "webhook_callback": config.app_host,
                "identifier": task,
            },
            headers={"Authorization": f"Bearer {config.docparser_token}"},
        )
        if response.status_code != 200:
            raise HTTPException(status_code=500, detail=response.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return JSONResponse(
        content={"message": "File ingested successfully", "task_id": task}
    )
