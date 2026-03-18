"""
File: /brains.py
Created Date: Monday December 1st 2025
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Wednesday March 4th 2026 9:35:41 pm
Modified By: Christian Nonis <alch.infoemail@gmail.com>
-----
"""

import json
import os
from email import policy
from email.parser import BytesParser

from fastapi import Request
from fastapi.responses import JSONResponse
from pymongo.errors import OperationFailure, ServerSelectionTimeoutError
from starlette.middleware.base import BaseHTTPMiddleware
from starlette import status

from src.services.data.main import data_adapter
from src.services.kg_agent.main import cache_adapter


def _brain_id_from_multipart(body: bytes, content_type: str) -> str | None:
    try:
        message_bytes = (
            f"Content-Type: {content_type}\r\n\r\n".encode("utf-8", errors="replace")
            + body
        )
        msg = BytesParser(policy=policy.HTTP).parsebytes(message_bytes)
        for part in msg.iter_parts():
            if part.get_content_disposition() != "form-data":
                continue
            name = part.get_param("name", header="content-disposition")
            if name == "brain_id":
                raw = part.get_content_text()
                return raw.rstrip() if raw else None
    except Exception:
        pass
    return None


class BrainMiddleware(BaseHTTPMiddleware):
    excluded_prefixes: set[str] = set()

    async def dispatch(self, request: Request, call_next):
        if any(request.url.path.startswith(p) for p in self.excluded_prefixes):
            return await call_next(request)

        async def _get_brain_id():
            brain_id = None

            brain_id = request.headers.get("X-Brain-ID")

            if brain_id:
                brain_id = brain_id.rstrip()

            if brain_id is None:
                brain_id = request.query_params.get("brain_id")
                if brain_id:
                    brain_id = brain_id.rstrip()

            if brain_id is None and request.method in ("POST", "PUT", "PATCH"):
                content_type = request.headers.get("content-type", "") or ""
                body = await request.body()
                if "application/json" in content_type and body:
                    try:
                        body_data = json.loads(body)
                        if isinstance(body_data, dict):
                            brain_id = body_data.get("brain_id")
                            if brain_id:
                                brain_id = brain_id.rstrip()
                    except (json.JSONDecodeError, ValueError):
                        pass
                elif "multipart/form-data" in content_type and body:
                    form_brain_id = _brain_id_from_multipart(body, content_type)
                    if form_brain_id:
                        brain_id = form_brain_id

                body_sent = [False]

                async def receive():
                    if not body_sent[0]:
                        body_sent[0] = True
                        return {"type": "http.request", "body": body, "more_body": False}
                    return {"type": "http.request", "body": b"", "more_body": False}

                request._receive = receive
            request.state.brain_id = brain_id
            return brain_id

        # Variables ----------------------------------------------
        brain_id = await _get_brain_id()
        brain_creation_allowed = os.getenv("BRAIN_CREATION_ALLOWED") == "true"
        default_brain_fallback = os.getenv("DEFAULT_BRAIN_FALLBACK") == "true"
        cached_brain_id = cache_adapter.get(key=f"brain:{brain_id}", brain_id="system")

        # Bypassing system routes --------------------------------
        if request.url.path.startswith("/system") or request.url.path == "/":
            return await call_next(request)

        # Cleanup checks -----------------------------------------
        if brain_id == "system":
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"detail": "System brain is not allowed to be used."},
            )
        if brain_id and not brain_id.isalnum():
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"detail": "Brain ID must be alphanumeric.", "value": brain_id},
            )

        try:
            if brain_id and not cached_brain_id:
                stored_brain = data_adapter.get_brain(name_key=brain_id)

                request.state.brain_id = brain_id

                if not stored_brain and brain_creation_allowed:
                    new_brain = data_adapter.create_brain(name_key=brain_id)
                    cache_adapter.set(
                        key=f"brain:{brain_id}",
                        value=new_brain.id,
                        brain_id="system",
                    )
                elif stored_brain:
                    cache_adapter.set(
                        key=f"brain:{brain_id}",
                        value=stored_brain.id,
                        brain_id="system",
                    )
            elif not brain_id and default_brain_fallback:
                default_brain = data_adapter.get_brain(name_key="default")
                if not default_brain:
                    default_brain = data_adapter.create_brain(name_key="default")
                    cache_adapter.set(
                        key="brain:default",
                        value=default_brain.id,
                        brain_id="system",
                    )
                    request.state.brain_id = "default"
                else:
                    cache_adapter.set(
                        key="brain:default",
                        value=default_brain.id,
                        brain_id="system",
                    )
                    request.state.brain_id = "default"
        except (OperationFailure, ServerSelectionTimeoutError) as e:
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={
                    "detail": "Database unavailable. Check MongoDB connection and credentials (e.g. MONGO_* or MONGO_CONNECTION_STRING).",
                    "error": str(e),
                },
            )

        if getattr(request.state, "brain_id", None) is None:
            return JSONResponse(
                status_code=status.HTTP_406_NOT_ACCEPTABLE,
                content={"detail": "Brain not found or creation is not allowed."},
            )

        return await call_next(request)
