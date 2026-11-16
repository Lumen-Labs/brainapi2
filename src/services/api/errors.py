"""Structured, agent-readable HTTP error responses."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, JsonValue
from starlette.exceptions import HTTPException as StarletteHTTPException

from src.lib.cypher_safety import InvalidCypherIdentifier


class ErrorInfo(BaseModel):
    """Stable machine-readable information attached to every API error."""

    code: str = Field(description="Stable error code for programmatic handling.")
    message: str = Field(description="Short human-readable error summary.")
    resolution: str = Field(description="Concrete guidance for resolving the error.")
    request_id: str = Field(description="Identifier shared with the X-Request-ID header.")


class ErrorResponse(BaseModel):
    """Additive error envelope that preserves FastAPI's legacy detail field."""

    detail: JsonValue = Field(description="Original FastAPI error detail value.")
    error: ErrorInfo


_STATUS_DEFAULTS: dict[int, tuple[str, str, str]] = {
    400: (
        "INVALID_REQUEST",
        "The request is invalid.",
        "Correct the request parameters and retry.",
    ),
    401: (
        "AUTH_INVALID",
        "Authentication failed.",
        "Send a valid BrainPAT or Bearer token for the selected brain.",
    ),
    403: (
        "FORBIDDEN",
        "The operation is not permitted.",
        "Use credentials with permission for this operation.",
    ),
    404: (
        "RESOURCE_NOT_FOUND",
        "The requested resource was not found.",
        "Check the path, identifier, and selected brain before retrying.",
    ),
    405: (
        "METHOD_NOT_ALLOWED",
        "The HTTP method is not allowed for this resource.",
        "Use one of the methods documented in the OpenAPI specification.",
    ),
    406: (
        "BRAIN_NOT_FOUND",
        "The selected brain could not be resolved.",
        "Create the brain or provide the correct X-Brain-ID value.",
    ),
    422: (
        "VALIDATION_ERROR",
        "Request validation failed.",
        "Correct the fields listed in detail and retry.",
    ),
    429: (
        "RATE_LIMITED",
        "The request rate limit was exceeded.",
        "Wait before retrying and use bounded exponential backoff.",
    ),
    500: (
        "INTERNAL_ERROR",
        "The service could not complete the request.",
        "Retry only when safe; contact support with the request ID if it persists.",
    ),
    501: (
        "not_implemented",
        "The operation is not implemented.",
        "Use a supported operation or consult the release notes for availability.",
    ),
    503: (
        "SERVICE_UNAVAILABLE",
        "A required service is temporarily unavailable.",
        "Retry with bounded exponential backoff after the dependency recovers.",
    ),
}


def _request_id(request: Request) -> str:
    return (
        request.headers.get("X-Request-ID")
        or request.headers.get("X-Trace-ID")
        or str(uuid4())
    )


def error_response(
    request: Request,
    *,
    status_code: int,
    detail: Any,
    code: str | None = None,
    message: str | None = None,
    resolution: str | None = None,
    extra: dict[str, Any] | None = None,
    headers: Mapping[str, str] | None = None,
) -> JSONResponse:
    """Create the common JSON envelope while retaining the legacy detail value."""

    default_code, default_message, default_resolution = _STATUS_DEFAULTS.get(
        status_code,
        _STATUS_DEFAULTS[500],
    )
    request_id = _request_id(request)
    content: dict[str, Any] = {
        "detail": detail,
        "error": {
            "code": code or default_code,
            "message": message or default_message,
            "resolution": resolution or default_resolution,
            "request_id": request_id,
        },
    }
    if extra:
        content.update(extra)
    response_headers = dict(headers or {})
    response_headers["X-Request-ID"] = request_id
    return JSONResponse(
        status_code=status_code,
        content=content,
        headers=response_headers,
    )


def install_error_handlers(app: FastAPI) -> None:
    """Install handlers for framework, validation, and unexpected exceptions."""

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException):
        return error_response(
            request,
            status_code=exc.status_code,
            detail=exc.detail,
            headers=exc.headers,
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request,
        exc: RequestValidationError,
    ):
        return error_response(
            request,
            status_code=422,
            detail=exc.errors(),
        )

    @app.exception_handler(InvalidCypherIdentifier)
    async def cypher_identifier_handler(
        request: Request,
        exc: InvalidCypherIdentifier,
    ):
        value = exc.value if isinstance(exc.value, str) and len(exc.value) <= 200 else None
        extra = {"kind": exc.kind}
        if value is not None:
            extra["value"] = value
        return error_response(
            request,
            status_code=400,
            detail=str(exc),
            code="CYPHER_IDENTIFIER_INVALID",
            message="A graph identifier in the request is invalid.",
            resolution=(
                "Use labels, relationship types, and brain IDs made of letters, "
                "digits, and underscore. Pass values as data, not as query syntax."
            ),
            extra=extra,
        )

    @app.exception_handler(Exception)
    async def unexpected_exception_handler(request: Request, _exc: Exception):
        return error_response(
            request,
            status_code=500,
            detail="Internal server error",
        )


COMMON_ERROR_RESPONSES = {
    400: {"model": ErrorResponse, "description": "Invalid request"},
    401: {"model": ErrorResponse, "description": "Authentication failed"},
    403: {"model": ErrorResponse, "description": "Operation forbidden"},
    404: {"model": ErrorResponse, "description": "Resource not found"},
    405: {"model": ErrorResponse, "description": "Method not allowed"},
    406: {"model": ErrorResponse, "description": "Brain not found"},
    422: {"model": ErrorResponse, "description": "Validation failed"},
    429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    500: {"model": ErrorResponse, "description": "Internal service error"},
    501: {"model": ErrorResponse, "description": "Operation not implemented"},
    503: {"model": ErrorResponse, "description": "Dependency unavailable"},
}
