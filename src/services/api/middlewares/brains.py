"""
File: /brains.py
Created Date: Monday December 1st 2025
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Monday December 1st 2025 9:06:30 pm
Modified By: the developer formerly known as Christian Nonis at <alch.infoemail@gmail.com>
-----
"""

import os
from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette import status

from src.services.data.main import data_adapter
from src.services.kg_agent.main import cache_adapter


class BrainMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        brain_id = None

        brain_id = request.query_params.get("brain_id")
        if brain_id:
            brain_id = brain_id.rstrip()

        if brain_id is None and request.method in ("POST", "PUT", "PATCH"):
            body = await request.body()
            if body:
                try:
                    import json

                    body_data = json.loads(body)
                    if isinstance(body_data, dict):
                        brain_id = body_data.get("brain_id")
                        if brain_id:
                            brain_id = brain_id.rstrip()
                except (json.JSONDecodeError, ValueError):
                    pass

            if body:

                async def receive():
                    return {"type": "http.request", "body": body}

                request._receive = receive

        if brain_id == "system":
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"detail": "System brain is not allowed to be used."},
            )
        if brain_id and not brain_id.isalnum():
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"detail": "Brain ID must be alphanumeric."},
            )

        if brain_id:
            cached_brain_id = cache_adapter.get(
                key=f"brain:{brain_id}", brain_id="system"
            )

            brain_creation_allowed = os.getenv("BRAIN_CREATION_ALLOWED") == "true"

            if not cached_brain_id and brain_creation_allowed:
                stored_brain = data_adapter.get_brain(name_key=brain_id)
                new_brain = None
                if not stored_brain:
                    new_brain = data_adapter.create_brain(name_key=brain_id)
                    cache_adapter.set(
                        key=f"brain:{brain_id}",
                        value=new_brain.id,
                        brain_id="system",
                    )
                else:
                    cache_adapter.set(
                        key=f"brain:{brain_id}",
                        value=stored_brain.id,
                        brain_id="system",
                    )

                cached_brain_id = new_brain.id if new_brain else stored_brain.id

                use_only_system_pat = os.getenv("USE_ONLY_SYSTEM_PAT") == "true"
                if not use_only_system_pat:
                    request.state.pat = new_brain.pat if new_brain else stored_brain.pat

            if not cached_brain_id:
                return JSONResponse(
                    status_code=status.HTTP_406_NOT_ACCEPTABLE,
                    content={"detail": "Brain not found or creation is not allowed."},
                )

        request.state.brain_id = brain_id

        return await call_next(request)
