import asyncio
import inspect
import logging
import os
import time
import traceback
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator, Optional

from src.lib.tracing.events import TraceEvent, TraceEventType, TraceSeverity
from src.lib.tracing.subscribers import TraceSubscriber

logger = logging.getLogger(__name__)

_trace_id_var: ContextVar[Optional[str]] = ContextVar("trace_id", default=None)
_tenant_id_var: ContextVar[Optional[str]] = ContextVar("tenant_id", default=None)


class LocalTraceQueue:
    def __init__(self, max_size: int = 10000):
        self._queue: asyncio.Queue[TraceEvent] = asyncio.Queue(maxsize=max_size)

    @property
    def size(self) -> int:
        return self._queue.qsize()

    def put_nowait(self, event: TraceEvent) -> bool:
        try:
            self._queue.put_nowait(event)
            return True
        except asyncio.QueueFull:
            return False

    async def get(self) -> TraceEvent:
        return await self._queue.get()

    def task_done(self) -> None:
        self._queue.task_done()

    def drain(self, limit: Optional[int] = None) -> list[TraceEvent]:
        events: list[TraceEvent] = []
        while not self._queue.empty() and (limit is None or len(events) < limit):
            events.append(self._queue.get_nowait())
            self._queue.task_done()
        return events


class TraceTracker:
    def __init__(
        self,
        *,
        queue: Optional[LocalTraceQueue] = None,
        max_queue_size: Optional[int] = None,
        slow_operation_ms: Optional[float] = None,
        downtime_ms: Optional[float] = None,
        expensive_loop_iterations: Optional[int] = None,
        enabled: Optional[bool] = None,
    ):
        self.queue = queue or LocalTraceQueue(
            max_queue_size
            if max_queue_size is not None
            else int(os.getenv("TRACE_QUEUE_MAX_SIZE", "10000"))
        )
        self.slow_operation_ms = (
            slow_operation_ms
            if slow_operation_ms is not None
            else float(os.getenv("TRACE_SLOW_OPERATION_MS", "1000"))
        )
        self.downtime_ms = (
            downtime_ms
            if downtime_ms is not None
            else float(os.getenv("TRACE_DOWNTIME_MS", "5000"))
        )
        self.expensive_loop_iterations = (
            expensive_loop_iterations
            if expensive_loop_iterations is not None
            else int(os.getenv("TRACE_EXPENSIVE_LOOP_ITERATIONS", "10000"))
        )
        self.enabled = (
            enabled
            if enabled is not None
            else os.getenv("TRACE_TRACKER_ENABLED", "true") == "true"
        )
        self._subscribers: list[TraceSubscriber] = []
        self._subscriber_tasks: list[asyncio.Task] = []

    @property
    def default_sla_ms(self) -> float:
        return self.slow_operation_ms

    def set_context(
        self, *, trace_id: Optional[str] = None, tenant_id: Optional[str] = None
    ):
        trace_token = _trace_id_var.set(trace_id) if trace_id is not None else None
        tenant_token = _tenant_id_var.set(tenant_id) if tenant_id is not None else None
        return trace_token, tenant_token

    def reset_context(self, tokens) -> None:
        trace_token, tenant_token = tokens
        if trace_token is not None:
            _trace_id_var.reset(trace_token)
        if tenant_token is not None:
            _tenant_id_var.reset(tenant_token)

    def subscribe(self, subscriber: TraceSubscriber) -> None:
        self._subscribers.append(subscriber)

    def clear_subscribers(self) -> None:
        self._subscribers.clear()

    def publish(
        self,
        event_type: TraceEventType | str,
        name: str,
        *,
        severity: TraceSeverity | str = TraceSeverity.INFO,
        tenant_id: Optional[str] = None,
        trace_id: Optional[str] = None,
        service: Optional[str] = None,
        operation: Optional[str] = None,
        duration_ms: Optional[float] = None,
        threshold_ms: Optional[float] = None,
        status_code: Optional[int] = None,
        message: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
        exception: Optional[BaseException] = None,
    ) -> Optional[TraceEvent]:
        if not self.enabled:
            return None
        event_trace_id = trace_id or _trace_id_var.get()
        event_message = message or (str(exception) if exception else None)
        event = TraceEvent(
            event_type=TraceEventType(event_type),
            severity=TraceSeverity(severity),
            name=name,
            service=service,
            operation=operation,
            tenant_id=tenant_id or _tenant_id_var.get(),
            trace_id=event_trace_id,
            duration_ms=duration_ms,
            threshold_ms=threshold_ms,
            status_code=status_code,
            error_type=type(exception).__name__ if exception else None,
            message=event_message,
            metadata=dict(metadata or {}),
            stack_trace=(
                "".join(traceback.format_exception(exception)) if exception else None
            ),
        )
        if not self.queue.put_nowait(event):
            logger.warning("Trace queue is full; dropping event %s", event.name)
            return None
        return event

    def exception(
        self,
        name: str,
        exception: BaseException,
        *,
        service: Optional[str] = None,
        operation: Optional[str] = None,
        tenant_id: Optional[str] = None,
        trace_id: Optional[str] = None,
        duration_ms: Optional[float] = None,
        threshold_ms: Optional[float] = None,
        status_code: Optional[int] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> Optional[TraceEvent]:
        return self.publish(
            TraceEventType.EXCEPTION,
            name,
            severity=TraceSeverity.ERROR,
            service=service,
            operation=operation,
            tenant_id=tenant_id,
            trace_id=trace_id,
            duration_ms=duration_ms,
            threshold_ms=threshold_ms,
            status_code=status_code,
            metadata=metadata,
            exception=exception,
        )

    def error(
        self,
        name: str,
        *,
        service: Optional[str] = None,
        operation: Optional[str] = None,
        tenant_id: Optional[str] = None,
        trace_id: Optional[str] = None,
        severity: TraceSeverity | str = TraceSeverity.ERROR,
        duration_ms: Optional[float] = None,
        threshold_ms: Optional[float] = None,
        status_code: Optional[int] = None,
        message: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> Optional[TraceEvent]:
        return self.publish(
            TraceEventType.ERROR,
            name,
            severity=severity,
            service=service,
            operation=operation,
            tenant_id=tenant_id,
            trace_id=trace_id,
            duration_ms=duration_ms,
            threshold_ms=threshold_ms,
            status_code=status_code,
            message=message,
            metadata=metadata,
        )

    def downtime(
        self,
        name: str,
        duration_ms: float,
        *,
        service: Optional[str] = None,
        operation: Optional[str] = None,
        tenant_id: Optional[str] = None,
        trace_id: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> Optional[TraceEvent]:
        return self.publish(
            TraceEventType.DOWNTIME,
            name,
            severity=TraceSeverity.CRITICAL,
            service=service,
            operation=operation,
            tenant_id=tenant_id,
            trace_id=trace_id,
            duration_ms=duration_ms,
            metadata=metadata,
        )

    def expensive_loop(
        self,
        name: str,
        iterations: int,
        *,
        service: Optional[str] = None,
        operation: Optional[str] = None,
        tenant_id: Optional[str] = None,
        trace_id: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
        threshold: Optional[int] = None,
    ) -> Optional[TraceEvent]:
        limit = (
            threshold
            if threshold is not None
            else self.expensive_loop_iterations
        )
        if iterations < limit:
            return None
        return self.publish(
            TraceEventType.EXPENSIVE_LOOP,
            name,
            severity=TraceSeverity.WARNING,
            service=service,
            operation=operation,
            tenant_id=tenant_id,
            trace_id=trace_id,
            metadata={**(metadata or {}), "iterations": iterations},
        )

    @contextmanager
    def span(
        self,
        name: str,
        *,
        tenant_id: Optional[str] = None,
        trace_id: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
        slow_operation_ms: Optional[float] = None,
        service: Optional[str] = None,
        operation: Optional[str] = None,
    ) -> Iterator[None]:
        tokens = self.set_context(trace_id=trace_id, tenant_id=tenant_id)
        started_at = time.perf_counter()
        try:
            yield
        except Exception as exc:
            duration_ms = (time.perf_counter() - started_at) * 1000
            self.exception(
                name,
                exc,
                service=service,
                operation=operation,
                metadata={**(metadata or {}), "duration_ms": duration_ms},
            )
            raise
        finally:
            duration_ms = (time.perf_counter() - started_at) * 1000
            threshold = (
                slow_operation_ms
                if slow_operation_ms is not None
                else self.slow_operation_ms
            )
            if duration_ms >= threshold:
                self.publish(
                    TraceEventType.SLA_BREACH,
                    name,
                    severity=TraceSeverity.WARNING,
                    service=service,
                    operation=operation,
                    duration_ms=duration_ms,
                    threshold_ms=threshold,
                    metadata=metadata,
                )
            self.reset_context(tokens)

    def track_loop(
        self,
        name: str,
        iterable,
        *,
        tenant_id: Optional[str] = None,
        trace_id: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
        threshold: Optional[int] = None,
    ):
        count = 0
        for item in iterable:
            count += 1
            yield item
        if count >= (threshold or self.expensive_loop_iterations):
            self.expensive_loop(
                name,
                count,
                tenant_id=tenant_id,
                trace_id=trace_id,
                metadata=metadata,
                threshold=threshold,
            )

    async def dispatch_once(self) -> TraceEvent:
        event = await self.queue.get()
        try:
            for subscriber in list(self._subscribers):
                result = subscriber.handle(event)
                if inspect.isawaitable(result):
                    await result
        finally:
            self.queue.task_done()
        return event

    async def subscribe_forever(self) -> None:
        while True:
            await self.dispatch_once()

    def start_subscribers(self) -> None:
        if not self._subscribers or self._subscriber_tasks:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._subscriber_tasks = [
            loop.create_task(self.subscribe_forever()) for _ in self._subscribers
        ]

    async def stop_subscribers(self) -> None:
        for task in self._subscriber_tasks:
            task.cancel()
        for task in self._subscriber_tasks:
            try:
                await task
            except asyncio.CancelledError:
                pass
        self._subscriber_tasks = []


trace_tracker = TraceTracker()
tracer = trace_tracker
