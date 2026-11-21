"""OpenAPI guarantees for discovery and LLM function-calling clients."""

from __future__ import annotations

import logging
import os
from typing import Any

from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi

logger = logging.getLogger("brainapi.openapi")

_HTTP_METHODS = {"get", "post", "put", "patch", "delete", "options", "head"}
_CORE_TAGS = {"public", "ingest", "retrieve", "model", "meta", "tasks", "system"}
_LIST_PATHS = {
    "/meta/entity-labels",
    "/meta/entity-properties",
    "/meta/relationships-properties",
    "/retrieve/hops",
    "/system/brains-list",
}
_NO_BRAIN_ID_PATHS = {"/health", "/demo/search", "/meta/login-info"}
_DEFAULT_SERVER_URL = "http://localhost:8000"
_X_BRAIN_ID = {
    "name": "X-Brain-ID",
    "in": "header",
    "required": False,
    "description": (
        "Alphanumeric brain identifier for brain-scoped operations. "
        "Send this header, or brain_id in the query string or JSON body, "
        "unless the request falls back to the default brain."
    ),
    "schema": {"type": "string", "pattern": "^[A-Za-z0-9]+$"},
}


def _error_response(description: str) -> dict[str, Any]:
    return {
        "description": description,
        "content": {
            "application/json": {
                "schema": {"$ref": "#/components/schemas/ErrorResponse"}
            }
        },
    }


def _generic_schema(model: str) -> dict[str, Any]:
    if model == "GenericListResponse":
        return {
            "type": "array",
            "description": "A list of operation-specific JSON values.",
            "items": {
                "oneOf": [
                    {"type": "string"},
                    {"type": "object", "additionalProperties": True},
                ]
            },
        }
    return {
        "type": "object",
        "description": (
            "An operation-specific JSON object from an installed extension."
        ),
        "additionalProperties": True,
    }


def _needs_brain_id(path: str) -> bool:
    if path in _NO_BRAIN_ID_PATHS or path == "/":
        return False
    if path.startswith("/system"):
        return False
    return True


def openapi_servers() -> list[dict[str, str]]:
    urls: list[str] = []
    multi = os.getenv("OPENAPI_SERVER_URLS", "").strip()
    if multi:
        urls.extend(part.strip() for part in multi.split(",") if part.strip())
    single = os.getenv("OPENAPI_SERVER_URL", "").strip()
    if single and single not in urls:
        urls.insert(0, single)
    if not urls:
        urls = [_DEFAULT_SERVER_URL]
    servers: list[dict[str, str]] = []
    for url in urls:
        description = "Local BrainAPI" if url == _DEFAULT_SERVER_URL else "BrainAPI"
        servers.append({"url": url, "description": description})
    return servers


def _build_openapi(app: FastAPI, routes: list[Any]) -> dict[str, Any]:
    return get_openapi(
        title=app.title,
        version=app.version,
        openapi_version=app.openapi_version,
        description=app.description,
        routes=routes,
    )


def _get_openapi_schema(app: FastAPI) -> dict[str, Any]:
    try:
        return _build_openapi(app, app.routes)
    except Exception:
        logger.exception("OpenAPI generation failed; retrying without failing routes")
        good: list[Any] = []
        for route in app.routes:
            try:
                _build_openapi(app, [route])
            except Exception:
                logger.exception(
                    "Skipping route during OpenAPI generation: %s",
                    getattr(route, "path", route),
                )
                continue
            good.append(route)
        try:
            return _build_openapi(app, good)
        except Exception:
            logger.exception("OpenAPI generation failed after skipping routes")
            return {
                "openapi": app.openapi_version,
                "info": {"title": app.title, "version": app.version},
                "paths": {},
            }


def _ensure_success_schema(
    path: str,
    method: str,
    operation: dict[str, Any],
    component_schemas: dict[str, Any],
) -> None:
    for status_code, response in operation.get("responses", {}).items():
        if not str(status_code).startswith("2") or not isinstance(response, dict):
            continue
        for media in response.get("content", {}).values():
            if not isinstance(media, dict) or media.get("schema") != {}:
                continue
            if _CORE_TAGS.intersection(operation.get("tags", [])):
                logger.warning(
                    "Core operation %s %s is missing a typed success response; using a generic schema",
                    method.upper(),
                    path,
                )
            model = (
                "GenericListResponse" if path in _LIST_PATHS else "GenericObjectResponse"
            )
            component_schemas.setdefault(model, _generic_schema(model))
            media["schema"] = {"$ref": f"#/components/schemas/{model}"}


def _ensure_brain_id_parameter(path: str, operation: dict[str, Any]) -> None:
    if not _needs_brain_id(path):
        return
    parameters = operation.setdefault("parameters", [])
    for parameter in parameters:
        if not isinstance(parameter, dict):
            continue
        if parameter.get("name") == "X-Brain-ID" and parameter.get("in") == "header":
            return
    parameters.append(dict(_X_BRAIN_ID))


def install_openapi_contract(app: FastAPI) -> None:
    """Install a deterministic, fully-described OpenAPI schema builder."""

    def custom_openapi() -> dict[str, Any]:
        if app.openapi_schema is not None:
            return app.openapi_schema

        schema = _get_openapi_schema(app)
        components = schema.setdefault("components", {})
        component_schemas = components.setdefault("schemas", {})
        security_schemes = components.setdefault("securitySchemes", {})
        security_schemes.update(
            {
                "BrainPAT": {
                    "type": "apiKey",
                    "in": "header",
                    "name": "BrainPAT",
                    "description": "BrainAPI personal access token.",
                },
                "BearerAuth": {
                    "type": "http",
                    "scheme": "bearer",
                    "description": "BrainAPI PAT sent as a Bearer token.",
                },
            }
        )

        protected_errors = {
            "400": _error_response("Invalid request"),
            "401": _error_response("Authentication failed"),
            "403": _error_response("Operation forbidden"),
            "404": _error_response("Resource not found"),
            "405": _error_response("Method not allowed"),
            "406": _error_response("Brain not found"),
            "422": _error_response("Validation failed"),
            "429": _error_response("Rate limit exceeded"),
            "500": _error_response("Internal service error"),
            "503": _error_response("Dependency unavailable"),
        }

        for path, path_item in schema.get("paths", {}).items():
            for method, operation in path_item.items():
                if method.lower() not in _HTTP_METHODS or not isinstance(operation, dict):
                    continue
                summary = operation.get("summary") or operation.get("operationId")
                if not operation.get("description"):
                    operation["description"] = (
                        f"{summary}. See the BrainAPI developer documentation for "
                        "request, response, authentication, and retry semantics."
                    )

                _ensure_success_schema(path, method, operation, component_schemas)
                _ensure_brain_id_parameter(path, operation)

                is_public = path in {"/health", "/demo/search"}
                if is_public:
                    operation["security"] = []
                    public_errors = {
                        "405": _error_response("Method not allowed"),
                        "500": _error_response("Internal service error"),
                        "503": _error_response("Dependency unavailable"),
                    }
                    if path == "/demo/search":
                        public_errors.update(
                            {
                                "404": _error_response("Public demo disabled"),
                                "422": _error_response("Validation failed"),
                            }
                        )
                    responses = operation.setdefault("responses", {})
                    for code, response in public_errors.items():
                        if code == "422" or code not in responses:
                            responses[code] = response
                else:
                    operation["security"] = [{"BrainPAT": []}, {"BearerAuth": []}]
                    responses = operation.setdefault("responses", {})
                    for code, response in protected_errors.items():
                        if code == "422" or code not in responses:
                            responses[code] = response

                if path == "/demo/search":
                    operation.setdefault("responses", {})["429"] = _error_response(
                        "Public demo rate limit exceeded"
                    )
                    operation["x-rate-limit"] = {
                        "requests": 30,
                        "window": "60 seconds",
                    }

        schema["servers"] = openapi_servers()
        app.openapi_schema = schema
        return schema

    app.openapi = custom_openapi
