"""
File: /ingest.py
Created Date: Sunday October 19th 2025
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Sunday October 19th 2025 12:41:30 pm
Modified By: the developer formerly known as Christian Nonis at <alch.infoemail@gmail.com>
-----
"""

import json
import asyncio
from fastapi import APIRouter, HTTPException
from typing import Literal
from typing_extensions import Annotated
from fastapi import Form, UploadFile
from starlette.responses import JSONResponse
from celery.exceptions import OperationalError

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
async def ingest_data(data: IngestionRequestBody):
    """
    Ingest data to the processing pipeline and save to the memory.
    """
    task = None
    for attempt in range(MAX_TASK_RETRIES):
        try:
            task = ingest_data_task.delay(data.model_dump())
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
    file: Annotated[UploadFile, Form()],
    brain_id: str = Form(default="default"),
    understanding_level: str = Annotated[
        Literal["basic", "deep"], Form(default="deep")
    ],
):
    """
    Ingest a file into the processing pipeline and save to the memory.
    """

    return JSONResponse(content={"message": "File ingested successfully"})
