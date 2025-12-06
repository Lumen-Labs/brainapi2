"""
File: /auth.py
Created Date: Thursday November 27th 2025
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Thursday November 27th 2025 10:20:03 pm
Modified By: the developer formerly known as Christian Nonis at <alch.infoemail@gmail.com>
-----
"""

import os
from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette import status

from src.services.kg_agent.main import cache_adapter
from src.services.data.main import data_adapter


class BrainPATMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.method == "OPTIONS":
            return await call_next(request)

        brainpat = request.headers.get("BrainPAT") or getattr(
            request.state, "pat", None
        )
        brain_id = getattr(request.state, "brain_id", None)

        cachepat_key = f"brainpat:{brain_id or 'default'}"

        use_only_system_pat = os.getenv("USE_ONLY_SYSTEM_PAT") == "true"

        if use_only_system_pat:
            cachepat_key = "brainpat:system"

        cached_brainpat = cache_adapter.get(key=cachepat_key, brain_id="system")

        if not cached_brainpat and not use_only_system_pat:
            stored_brain = data_adapter.get_brain(name_key=brain_id)
            system_pat = os.getenv("BRAINPAT_TOKEN")
            if not stored_brain:
                if brainpat != system_pat:
                    return JSONResponse(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        content={"detail": "Invalid or missing BrainPAT header"},
                    )
                cached_brainpat = system_pat
            else:
                cached_brainpat = stored_brain.pat
                cache_adapter.set(
                    key=cachepat_key,
                    value=stored_brain.pat,
                    brain_id="system",
                )
        if not cached_brainpat and use_only_system_pat:
            system_pat = os.getenv("BRAINPAT_TOKEN")
            cached_brainpat = system_pat
            cache_adapter.set(
                key="brainpat:system",
                value=system_pat,
                brain_id="system",
            )

        if cached_brainpat != brainpat:
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"detail": "Invalid or missing BrainPAT header"},
            )
        response = await call_next(request)

        return response
