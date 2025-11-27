"""
File: /auth.py
Created Date: Thursday November 27th 2025
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Thursday November 27th 2025 10:20:03 pm
Modified By: the developer formerly known as Christian Nonis at <alch.infoemail@gmail.com>
-----
"""

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette import status

from src.config import config


class BrainPATMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.method == "OPTIONS":
            return await call_next(request)
        brainpat = request.headers.get("BrainPAT")
        if brainpat != config.brainpat_token:
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"detail": "Invalid or missing BrainPAT header"},
            )
        response = await call_next(request)
        return response
