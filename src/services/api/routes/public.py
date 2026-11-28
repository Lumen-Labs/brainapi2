"""Public, read-only API surfaces intended for discovery and evaluation."""

from __future__ import annotations

import os
import re

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.config import config
from src.constants.data import BRAIN_VERSION
from src.services.api.constants.requests import SearchRequestBody, SearchResponse
from src.services.api.controllers.search import search as search_controller
from src.services.api.errors import ErrorResponse


def running_build_id() -> str:
    for key in ("BUILD_SHA", "GIT_SHA"):
        value = (os.getenv(key) or "").strip()
        if value:
            return value
    return "unknown"


def running_version() -> str:
    for key in ("BRAINAPI_VERSION", "BUILD_VERSION"):
        value = (os.getenv(key) or "").strip().lstrip("v")
        if value:
            match = re.match(r"(\d+\.\d+\.\d+)", value)
            return match.group(1) if match else value
    image = (os.getenv("BRAINAPI_IMAGE") or "").strip()
    if image:
        tag = image.rsplit(":", 1)[-1]
        match = re.match(r"v?(\d+\.\d+\.\d+)", tag)
        if match:
            return match.group(1)
    return BRAIN_VERSION


def probe_data_store() -> None:
    from src.services.data.main import data_adapter

    ping = getattr(data_adapter, "ping", None)
    if not callable(ping):
        raise RuntimeError("Data store health check is unavailable")
    ping()


class HealthResponse(BaseModel):
    status: str = Field(description="Service health status.")
    service: str = Field(description="Service name.")
    version: str = Field(description="Running BrainAPI version.")
    build: str = Field(
        default="unknown",
        description="Commit SHA or build ID for this image.",
    )


PUBLIC_ERROR_RESPONSES = {
    404: {"model": ErrorResponse, "description": "Public demo is disabled"},
    422: {"model": ErrorResponse, "description": "Invalid query"},
    500: {"model": ErrorResponse, "description": "Internal service error"},
    503: {"model": ErrorResponse, "description": "Demo search is unavailable"},
}

public_router = APIRouter(tags=["public"])


@public_router.get(
    "/health",
    response_model=HealthResponse,
    responses={
        503: {"model": HealthResponse, "description": "A required database is down"},
    },
    openapi_extra={"security": []},
)
async def health() -> HealthResponse | JSONResponse:
    """Return service health, including a cheap database ping."""

    payload = {
        "service": "brainapi",
        "version": running_version(),
        "build": running_build_id(),
    }
    try:
        probe_data_store()
    except Exception:
        return JSONResponse(
            status_code=503,
            content={**payload, "status": "degraded"},
        )
    return HealthResponse(status="ok", **payload)


@public_router.get(
    "/demo/search",
    response_model=SearchResponse,
    responses=PUBLIC_ERROR_RESPONSES,
    openapi_extra={"security": []},
)
async def demo_search(
    query: str = Query(
        ...,
        min_length=1,
        max_length=500,
        description="Search query over the public BrainAPI documentation corpus.",
    ),
    k: int = Query(
        10,
        ge=1,
        le=10,
        description="Number of public documentation passages to return (maximum 10).",
    ),
) -> SearchResponse:
    """Search the fixed public demo brain using passages-only BM25 retrieval.

    This endpoint never accepts a caller-selected brain and exposes no write path.
    Production applies an additional per-client rate limit at the edge.
    """

    if os.getenv("PUBLIC_DEMO_ENABLED", "false").strip().lower() != "true":
        raise HTTPException(status_code=404, detail="Public demo search is disabled")

    brain_id = os.getenv("PUBLIC_DEMO_BRAIN_ID", "agentdemo").strip()
    if not brain_id or not brain_id.isalnum() or brain_id == "system":
        raise HTTPException(
            status_code=503,
            detail="Public demo brain is not configured correctly",
        )

    try:
        configured_max = int(os.getenv("PUBLIC_DEMO_MAX_K", "10"))
    except ValueError as exc:
        raise HTTPException(
            status_code=503,
            detail="PUBLIC_DEMO_MAX_K must be an integer",
        ) from exc
    max_k = min(max(configured_max, 1), 10)
    if k > max_k:
        raise HTTPException(
            status_code=422,
            detail=f"k must be less than or equal to {max_k}",
        )

    if (
        not config.search_enabled
        or not config.search_use_bm25
        or config.search_use_dense
    ):
        raise HTTPException(
            status_code=503,
            detail="Public demo requires BM25-only Search configuration",
        )

    return await search_controller(
        SearchRequestBody(
            query=query,
            k=k,
            channels=["passages"],
            brain_id=brain_id,
        )
    )
