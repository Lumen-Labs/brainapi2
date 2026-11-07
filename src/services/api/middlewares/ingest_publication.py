"""Hold brain access until the downstream ASGI response has finished."""

import os

import anyio
from starlette.concurrency import run_in_threadpool
from starlette.responses import JSONResponse

from src.core.saving.ingest_access_gate import BrainPublicationBusy, brain_access


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
