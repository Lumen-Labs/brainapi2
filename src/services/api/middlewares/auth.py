"""
File: /auth.py
Created Date: Thursday November 27th 2025
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Thursday February 19th 2026 7:45:12 pm
Modified By: Christian Nonis <alch.infoemail@gmail.com>
-----
"""

import json
import os
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette import status

from src.services.api.console_static import is_console_path
from src.services.api.errors import error_response
from src.services.api.middlewares.routing import request_matches_registered_route
from src.services.kg_agent.main import cache_adapter
from src.services.data.main import data_adapter
from src.utils.tokens import pat_matches


class BrainPATMiddleware(BaseHTTPMiddleware):
    excluded_prefixes: set[str] = {"/console", "/docs", "/redoc", "/demo"}
    auth_exempt_paths: set[str] = {
        "/health",
        "/openapi.json",
        "/meta/login-info",
    }

    async def dispatch(self, request: Request, call_next):
        if request.method == "OPTIONS":
            return await call_next(request)

        if is_console_path(request.url.path):
            return await call_next(request)

        if any(request.url.path.startswith(p) for p in self.excluded_prefixes):
            return await call_next(request)

        if request.url.path in self.auth_exempt_paths:
            return await call_next(request)

        if not request_matches_registered_route(request):
            return await call_next(request)

        brainpat = request.headers.get("BrainPAT") or getattr(
            request.state, "pat", None
        )
        if not brainpat:
            brainpat = request.headers.get("Authorization")
            if brainpat:
                scheme, _, token = brainpat.partition(" ")
                brainpat = token.rstrip() if scheme.lower() == "bearer" else None
        if not brainpat:
            return error_response(
                request,
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or missing BrainPAT header",
            )
        system_pat = os.getenv("BRAINPAT_TOKEN")
        if request.url.path.startswith("/system") or request.url.path == "/":
            if pat_matches(brainpat, system_pat):
                return await call_next(request)
            return error_response(
                request,
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or missing BrainPAT header",
            )

        brain_id = getattr(request.state, "brain_id", None)
        if not brain_id:
            return error_response(
                request,
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Brain ID is required.",
                code="BRAIN_ID_REQUIRED",
                message="A brain identifier is required.",
                resolution="Send X-Brain-ID with the intended alphanumeric brain name.",
            )
        cachepat_key = f"brainpat:{brain_id}"
        scope_cache_key = f"brainpat-scopes:{brain_id}"
        if pat_matches(brainpat, system_pat):
            from src.core.auth.scopes import env_system_pat_scopes, principal_for_system

            request.state.principal = principal_for_system()
            request.state.scopes = env_system_pat_scopes()
            return await call_next(request)
        from src.core.auth.scopes import effective_pat_scopes, principal_for_brain

        stored_brain = None
        scopes = None
        try:
            cached_brainpat = cache_adapter.get(key=cachepat_key, brain_id="system")

            if not cached_brainpat:
                stored_brain = data_adapter.get_brain(name_key=brain_id)
                if not stored_brain or not pat_matches(brainpat, stored_brain.pat):
                    return error_response(
                        request,
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="Invalid or missing BrainPAT header",
                    )
                cache_adapter.set(
                    key=cachepat_key, value=stored_brain.pat, brain_id="system"
                )
                scopes = effective_pat_scopes(stored_brain)
                cache_adapter.set(
                    key=scope_cache_key,
                    value=json.dumps(scopes),
                    brain_id="system",
                    expires_in=30,
                )
            elif not pat_matches(brainpat, cached_brainpat):
                return error_response(
                    request,
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid or missing BrainPAT header",
                )
            else:
                cached_scopes = cache_adapter.get(key=scope_cache_key, brain_id="system")
                if cached_scopes:
                    try:
                        scopes = json.loads(cached_scopes)
                    except (TypeError, ValueError):
                        scopes = None
                if scopes is None:
                    stored_brain = data_adapter.get_brain(name_key=brain_id)
                    scopes = (
                        effective_pat_scopes(stored_brain) if stored_brain else []
                    )
                    cache_adapter.set(
                        key=scope_cache_key,
                        value=json.dumps(scopes),
                        brain_id="system",
                        expires_in=30,
                    )
        except Exception:
            return error_response(
                request,
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Authentication store unavailable",
            )

        request.state.principal = principal_for_brain(brain_id)
        request.state.scopes = scopes or []

        response = await call_next(request)

        return response
