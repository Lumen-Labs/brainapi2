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

        # Variables ----------------------------------------------
        brainpat = request.headers.get("BrainPAT") or getattr(
            request.state, "pat", None
        )
        system_pat = os.getenv("BRAINPAT_TOKEN")

        if request.url.path.startswith("/system") or request.url.path == "/":
            if brainpat == system_pat:
                return await call_next(request)
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"detail": "Invalid or missing BrainPAT header"},
            )

        brain_id = getattr(request.state, "brain_id", None)
        if not brain_id:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"detail": "Brain ID is required."},
            )
        cachepat_key = f"brainpat:{brain_id}"
        cached_brainpat = cache_adapter.get(key=cachepat_key, brain_id="system")

        # Logic --------------------------------------------------
        if brainpat == system_pat:
            return await call_next(request)

        if not cached_brainpat:
            stored_brain = data_adapter.get_brain(name_key=brain_id)
            if not stored_brain or stored_brain.pat != brainpat:
                return JSONResponse(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    content={"detail": "Invalid or missing BrainPAT header"},
                )
            cached_brainpat = stored_brain.pat
            cache_adapter.set(
                key=cachepat_key, value=stored_brain.pat, brain_id="system"
            )
        elif cached_brainpat != brainpat:
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"detail": "Invalid or missing BrainPAT header"},
            )

        response = await call_next(request)

        return response
