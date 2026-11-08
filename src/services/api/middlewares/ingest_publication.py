"""Hold brain access until the downstream ASGI response has finished."""

import os

import anyio
from starlette.concurrency import run_in_threadpool
from starlette.responses import JSONResponse

from src.config import config
from src.core.saving import ingest_access_gate
from src.core.saving.ingest_access_gate import BrainPublicationBusy, brain_access
from src.lib.postgresql._provisioning import bind_brain_transaction, open_brain_read_snapshot


_EXPLORE_READS = {
    "/meta/summary",
    "/meta/entity-labels",
    "/meta/entity-properties",
    "/meta/relationships-properties",
    "/retrieve/entities",
    "/retrieve/entities/neighbors",
    "/retrieve/relationships",
    "/retrieve/hops",
    "/retrieve/text-chunks",
}


def _read_atomic_gate(transaction):
    with bind_brain_transaction(transaction):
        gate = ingest_access_gate._gates(transaction.brain_id).find_one({"_id": "publication"})
        return bool(gate and gate.get("publication") and gate.get("atomic"))


class IngestPublicationMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        brain_id = scope.get("state", {}).get("brain_id")
        path = scope.get("path", "")
        task_path = path.startswith("/tasks/")
        task_control = task_path and (
            scope.get("method") == "GET" or path.endswith("/cancel")
        )
        if (
            scope["type"] != "http"
            or not brain_id
            or task_control
            or os.getenv("STAGED_INGESTION_ENABLED") != "1"
        ):
            await self.app(scope, receive, send)
            return

        access = brain_access(brain_id)
        try:
            await run_in_threadpool(access.__enter__)
        except BrainPublicationBusy:
            explore_read = scope.get("method") == "GET" and (
                path in _EXPLORE_READS or path.startswith("/retrieve/explore/v1/")
            )
            if (
                explore_read
                and os.getenv("INGEST_POSTGRES_ATOMIC_PUBLICATION") == "1"
                and config.data_db == config.vector_db == "postgresql"
                and config.graph_db == "networkx"
            ):
                if await self._serve_atomic_read(brain_id, scope, receive, send):
                    return
            response = JSONResponse(
                {"detail": "Brain data is being published. Try again shortly."},
                status_code=503,
                headers={"Retry-After": "2"},
            )
            await response(scope, receive, send)
            return

        try:
            await self.app(scope, receive, send)
        finally:
            with anyio.CancelScope(shield=True):
                await run_in_threadpool(access.__exit__, None, None, None)

    async def _serve_atomic_read(self, brain_id, scope, receive, send):
        snapshot = open_brain_read_snapshot(brain_id)
        transaction = await run_in_threadpool(snapshot.__enter__)
        try:
            # Check the marker in the same snapshot used by the read. A legacy
            # publisher may have taken over between admission and this query.
            if not await run_in_threadpool(_read_atomic_gate, transaction):
                return False
            with bind_brain_transaction(transaction):
                await self.app(scope, receive, send)
            return True
        finally:
            with anyio.CancelScope(shield=True):
                await run_in_threadpool(snapshot.__exit__, None, None, None)
