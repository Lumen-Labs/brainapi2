"""Versioned, authenticated progressive exploration endpoints."""

from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from src.config import config
from src.services.api.dependencies import get_brain_id
from src.services.explore.store import InvalidCursor, SnapshotExpired, atlas_store

explore_router = APIRouter(prefix="/explore/v1", tags=["explore"])


class AtlasReference(BaseModel):
    id: str
    name: str


class AtlasRepresentative(AtlasReference):
    types: list[str]


class AtlasGroup(AtlasRepresentative):
    parentId: str | None
    count: int
    representatives: list[AtlasRepresentative]


class AtlasLink(BaseModel):
    id: str
    fromId: str
    toId: str


class AtlasGroupLink(AtlasLink):
    count: int


DetailValue = str | int | float | bool | None


class AtlasDetails(BaseModel):
    description: str | None
    details: dict[str, DetailValue | list[DetailValue]]
    updatedAt: str | None
    advanced: dict[str, DetailValue]


class AtlasThing(AtlasRepresentative, AtlasDetails):
    typeLabels: list[str]


class AtlasConnection(AtlasLink, AtlasDetails):
    type: str
    typeLabel: str
    direction: Literal["in", "out", "neutral"]


class AtlasTotals(BaseModel):
    things: int | None = None
    connections: int | None = None


class AtlasCoverage(AtlasTotals):
    aggregated: bool


class AtlasResponse(BaseModel):
    version: Literal[1]
    snapshotId: str | None = None
    builtAt: str | None = None
    status: Literal["ready", "building", "refreshing", "failed"]
    groups: list[AtlasGroup] = Field(default_factory=list)
    groupLinks: list[AtlasGroupLink] = Field(default_factory=list)
    groupLinksTruncated: bool = False
    things: list[AtlasThing] = Field(default_factory=list)
    connections: list[AtlasConnection] = Field(default_factory=list)
    nextCursor: str | None = None
    totals: AtlasTotals = Field(default_factory=AtlasTotals)
    ancestors: list[AtlasReference] = Field(default_factory=list)
    coverage: AtlasCoverage | None = None


def filters(value):
    if not value:
        return []
    result = sorted({s.strip() for s in value.split(",") if s.strip()})
    if len(result) > 50 or any(len(s) > 128 for s in result):
        raise HTTPException(422, detail="Invalid filters")
    return result


async def read(brain_id, **kwargs):
    if str(config.graph_db).lower() not in ("postgresql", "networkx"):
        raise HTTPException(
            501, detail="Progressive exploration requires the PostgreSQL graph backend"
        )
    try:
        return await run_in_threadpool(atlas_store().read, brain_id, **kwargs)
    except SnapshotExpired:
        raise HTTPException(
            409, detail="The overview has expired. Return to the overview to refresh."
        ) from None
    except InvalidCursor:
        raise HTTPException(422, detail="Invalid exploration cursor") from None
    except KeyError:
        raise HTTPException(404, detail="Entity not found") from None


@explore_router.get("/overview", response_model=AtlasResponse)
async def overview(
    brain_id: str = Depends(get_brain_id),
    types: str = Query("", max_length=6500),
    connectionTypes: str = Query("", max_length=6500),
):
    return await read(
        brain_id, types=filters(types), connection_types=filters(connectionTypes)
    )


@explore_router.get("/communities/{community_id}", response_model=AtlasResponse)
async def community(
    community_id: str,
    snapshot: str = Query(..., max_length=64),
    cursor: str = Query("", max_length=2048),
    types: str = Query("", max_length=6500),
    connectionTypes: str = Query("", max_length=6500),
    brain_id: str = Depends(get_brain_id),
):
    return await read(
        brain_id,
        group=community_id,
        snapshot=snapshot,
        cursor=cursor,
        types=filters(types),
        connection_types=filters(connectionTypes),
    )


@explore_router.get("/neighbors/{entity_id}", response_model=AtlasResponse)
async def neighbors(
    entity_id: str,
    cursor: str = Query("", max_length=2048),
    types: str = Query("", max_length=6500),
    connectionTypes: str = Query("", max_length=6500),
    brain_id: str = Depends(get_brain_id),
):
    return await read(
        brain_id,
        node=entity_id,
        cursor=cursor,
        types=filters(types),
        connection_types=filters(connectionTypes),
    )
