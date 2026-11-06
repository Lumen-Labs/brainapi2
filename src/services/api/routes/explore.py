"""Versioned, authenticated progressive exploration endpoints."""

from fastapi import APIRouter, Depends, HTTPException, Query
from starlette.concurrency import run_in_threadpool

from src.config import config
from src.services.api.dependencies import get_brain_id
from src.services.explore.store import InvalidCursor, SnapshotExpired, atlas_store

explore_router = APIRouter(prefix="/explore/v1", tags=["explore"])


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


@explore_router.get("/overview")
async def overview(
    brain_id: str = Depends(get_brain_id),
    types: str = Query("", max_length=6500),
    connectionTypes: str = Query("", max_length=6500),
):
    return await read(
        brain_id, types=filters(types), connection_types=filters(connectionTypes)
    )


@explore_router.get("/communities/{community_id}")
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


@explore_router.get("/neighbors/{entity_id}")
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
