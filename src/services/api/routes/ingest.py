"""
File: /ingest.py
Created Date: Sunday October 19th 2025
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Sunday October 19th 2025 12:41:30 pm
Modified By: the developer formerly known as Christian Nonis at <alch.infoemail@gmail.com>
-----
"""

from fastapi import APIRouter
from starlette.responses import JSONResponse

from src.services.api.constants.requests import (
    IngestionRequestBody,
    IngestionStructuredRequestBody,
)
from src.workers.tasks.ingestion import ingest_data as ingest_data_task
from src.workers.tasks.ingestion import (
    ingest_structured_data as ingest_structured_data_task,
)

ingest_router = APIRouter(prefix="/ingest", tags=["ingest"])


@ingest_router.post(path="/")
async def ingest_data(data: IngestionRequestBody):
    """
    Ingest data to the processing pipeline and save to the memory.
    """
    task = ingest_data_task.delay(data.model_dump())

    return JSONResponse(
        content={"message": "Data ingested successfully", "task_id": task.id}
    )


@ingest_router.post(path="/structured")
async def ingest_structured_data(data: IngestionStructuredRequestBody):
    """
    Ingest structured data to the processing pipeline and save to the memory.
    """
    print("[data]", data)
    task = ingest_structured_data_task.delay(data.model_dump())

    return JSONResponse(
        content={"message": "Structured data ingested successfully", "task_id": task.id}
    )
