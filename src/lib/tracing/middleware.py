import os
import time
from uuid import uuid4

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from src.lib.tracing.events import TraceEventType, TraceSeverity
from src.lib.tracing.tracker import tracer


class TraceMiddleware:
    def __init__(
        self,
        app: ASGIApp,
        service_name: str,
        slow_request_ms: float | None = None,
    ):
        self.app = app
        self.service_name = service_name
        self.slow_request_ms = (
            slow_request_ms
            if slow_request_ms is not None
            else float(os.getenv("TRACE_SLOW_REQUEST_MS", "1000"))
        )

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return

        started_at = time.perf_counter()
        status_code: int | None = None
        tenant_id = self._tenant_id(scope)
        trace_id = self._header(scope, "x-trace-id") or str(uuid4())
        tokens = tracer.set_context(trace_id=trace_id, tenant_id=tenant_id)

        async def send_wrapper(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message.get("status")
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        except Exception as exc:
            duration_ms = (time.perf_counter() - started_at) * 1000
            tracer.exception(
                name=self._operation(scope),
                exception=exc,
                service=self.service_name,
                operation=self._operation(scope),
                duration_ms=duration_ms,
                status_code=status_code,
                threshold_ms=self.slow_request_ms,
                metadata=self._scope_metadata(scope, status_code),
            )
            tracer.reset_context(tokens)
            raise

        duration_ms = (time.perf_counter() - started_at) * 1000
        operation = self._operation(scope)
        if status_code is not None and status_code >= 500:
            tracer.error(
                name=operation,
                service=self.service_name,
                operation=operation,
                message=f"{self.service_name} returned HTTP {status_code}",
                severity=TraceSeverity.ERROR,
                duration_ms=duration_ms,
                status_code=status_code,
                threshold_ms=self.slow_request_ms,
                metadata=self._scope_metadata(scope, status_code),
            )

        if self.slow_request_ms is not None and duration_ms >= self.slow_request_ms:
            tracer.publish(
                TraceEventType.SLA_BREACH,
                name=operation,
                severity=TraceSeverity.WARNING,
                service=self.service_name,
                operation=operation,
                duration_ms=duration_ms,
                threshold_ms=self.slow_request_ms,
                status_code=status_code,
                message=f"{operation} exceeded SLA threshold",
                metadata=self._scope_metadata(scope, status_code),
            )
        tracer.reset_context(tokens)

    def _operation(self, scope: Scope) -> str:
        method = scope.get("method", scope["type"])
        return f"{method} {scope.get('path', '')}".strip()

    def _scope_metadata(self, scope: Scope, status_code: int | None) -> dict[str, str]:
        metadata = {
            "service": self.service_name,
            "protocol": scope["type"],
            "path": str(scope.get("path", "")),
        }
        if status_code is not None:
            metadata["status_code"] = str(status_code)
        return metadata

    def _tenant_id(self, scope: Scope) -> str | None:
        return (
            self._header(scope, "x-brain-id")
            or self._query_param(scope, "brain_id")
            or self._header(scope, "x-tenant-id")
        )

    def _header(self, scope: Scope, name: str) -> str | None:
        encoded_name = name.lower().encode()
        for header_name, value in scope.get("headers", []):
            if header_name.lower() == encoded_name:
                return value.decode(errors="replace").rstrip() or None
        return None

    def _query_param(self, scope: Scope, name: str) -> str | None:
        raw_query = scope.get("query_string", b"").decode(errors="replace")
        for item in raw_query.split("&"):
            key, _, value = item.partition("=")
            if key == name and value:
                return value.rstrip()
        return None
