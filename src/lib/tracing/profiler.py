import logging
import os
import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Iterator, Optional

from src.lib.tracing.events import TraceEventType
from src.lib.tracing.tracker import tracer

STAGE_PROFILER_ENV_FLAG = "TRACE_STAGE_PROFILER_ENABLED"

_profiler_var: ContextVar[Optional["StageProfiler"]] = ContextVar(
    "stage_profiler", default=None
)
_stack_var: ContextVar[tuple[str, ...]] = ContextVar(
    "stage_profiler_stack", default=()
)


@dataclass(frozen=True)
class StageSpan:
    name: str
    parent: Optional[str]
    started_ms: float
    ended_ms: float
    thread_id: int
    cpu_ms: Optional[float] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def wall_ms(self) -> float:
        return self.ended_ms - self.started_ms


class StageProfiler:
    """
    Wall-clock and CPU accounting for the named stages of a single call.

    Spans are recorded from whichever thread runs them, so a stage dispatched
    through `asyncio.to_thread` is attributed to that worker thread rather than
    to the event loop. `cpu_ms` is per-thread CPU time and is only collected for
    spans declared blocking, i.e. spans whose body contains no `await`: for a
    span that awaits, the loop thread's CPU clock also runs for every other
    coroutine it services, so the number would not belong to the stage.
    """

    def __init__(self, name: str, *, loop_thread_id: Optional[int] = None):
        self.name = name
        self.loop_thread_id = (
            loop_thread_id if loop_thread_id is not None else threading.get_ident()
        )
        self._origin = time.perf_counter()
        self._spans: list[StageSpan] = []
        self._lock = threading.Lock()
        self.last_report: Optional[dict[str, Any]] = None

    @property
    def spans(self) -> list[StageSpan]:
        with self._lock:
            return list(self._spans)

    def mark(self) -> float:
        return (time.perf_counter() - self._origin) * 1000

    def record(self, span: StageSpan) -> None:
        with self._lock:
            self._spans.append(span)

    def report(self) -> dict[str, Any]:
        spans = self.spans
        grouped: dict[tuple[str, Optional[str]], list[StageSpan]] = {}
        for span in spans:
            grouped.setdefault((span.name, span.parent), []).append(span)

        stages: list[dict[str, Any]] = []
        for (name, parent), group in grouped.items():
            intervals = [(s.started_ms, s.ended_ms) for s in group]
            wall_sum = sum(s.wall_ms for s in group)
            wall_union = _union_ms(intervals)
            cpu_ms = (
                sum(s.cpu_ms for s in group)
                if all(s.cpu_ms is not None for s in group)
                else None
            )
            stage: dict[str, Any] = {
                "stage": name,
                "parent": parent,
                "calls": len(group),
                "wall_ms": round(wall_union, 3),
                "wall_sum_ms": round(wall_sum, 3),
                "overlap_ms": round(max(0.0, wall_sum - wall_union), 3),
                "cpu_ms": None if cpu_ms is None else round(cpu_ms, 3),
                "io_wait_ms": (
                    None if cpu_ms is None else round(max(0.0, wall_sum - cpu_ms), 3)
                ),
                "on_loop": all(s.thread_id == self.loop_thread_id for s in group),
                "threads": len({s.thread_id for s in group}),
                "started_ms": round(min(i[0] for i in intervals), 3),
                "ended_ms": round(max(i[1] for i in intervals), 3),
            }
            merged_metadata = _merge_metadata(group)
            if merged_metadata:
                stage["detail"] = merged_metadata
            stages.append(stage)

        stages.sort(key=lambda item: item["started_ms"])
        loop_blocking = [
            (s.started_ms, s.ended_ms)
            for s in spans
            if s.cpu_ms is not None and s.thread_id == self.loop_thread_id
        ]
        self.last_report = {
            "profile": self.name,
            "total_ms": round(self.mark(), 3),
            "loop_blocked_ms": round(_union_ms(loop_blocking), 3),
            "span_count": len(spans),
            "stages": stages,
        }
        return self.last_report


def stage_profiling_enabled(requested: Optional[bool] = None) -> bool:
    if os.getenv(STAGE_PROFILER_ENV_FLAG, "false") == "true":
        return True
    return bool(requested)


def log_stage_report(
    report: Optional[dict[str, Any]],
    log: logging.Logger,
    *,
    level: int = logging.DEBUG,
) -> None:
    """Emit a compact per-stage timing line. No query text, ids, or secrets."""
    if not report:
        return
    parts = []
    for item in report.get("stages") or []:
        name = item.get("stage")
        if not name:
            continue
        parts.append(f"{name}={item.get('wall_ms')}ms/{item.get('calls')}")
    log.log(
        level,
        "stage timings profile=%s total_ms=%s loop_blocked_ms=%s %s",
        report.get("profile"),
        report.get("total_ms"),
        report.get("loop_blocked_ms"),
        " ".join(parts),
    )


@contextmanager
def profile_request(
    name: str,
    *,
    enabled: Optional[bool] = None,
    publish: bool = True,
) -> Iterator[Optional[StageProfiler]]:
    """
    Open a stage-profiling scope for one call, or yield None when disabled.

    Enabled per request by the caller, or globally by the
    `TRACE_STAGE_PROFILER_ENABLED` environment flag.
    """
    if not stage_profiling_enabled(enabled):
        yield None
        return

    profiler = StageProfiler(name)
    profiler_token = _profiler_var.set(profiler)
    stack_token = _stack_var.set((name,))
    try:
        yield profiler
    finally:
        _stack_var.reset(stack_token)
        _profiler_var.reset(profiler_token)
        report = profiler.report()
        if publish:
            tracer.publish(
                TraceEventType.LATENCY,
                f"{name}.stages",
                operation=name,
                duration_ms=report["total_ms"],
                metadata=report,
            )


@contextmanager
def profile_stage(
    name: str,
    *,
    blocking: bool = True,
    **metadata: Any,
) -> Iterator[dict[str, Any]]:
    """
    Time one stage, yielding a dict the body may extend with attribution detail.

    Pass `blocking=False` for a span whose body awaits; CPU time is then left
    unrecorded instead of being credited to the stage.
    """
    profiler = _profiler_var.get()
    if profiler is None:
        yield {}
        return

    detail: dict[str, Any] = dict(metadata)
    stack = _stack_var.get()
    stack_token = _stack_var.set(stack + (name,))
    started_ms = profiler.mark()
    cpu_started = time.thread_time() if blocking else None
    try:
        yield detail
    finally:
        ended_ms = profiler.mark()
        cpu_ms = (
            (time.thread_time() - cpu_started) * 1000
            if cpu_started is not None
            else None
        )
        _stack_var.reset(stack_token)
        profiler.record(
            StageSpan(
                name=name,
                parent=stack[-1] if stack else None,
                started_ms=started_ms,
                ended_ms=ended_ms,
                thread_id=threading.get_ident(),
                cpu_ms=cpu_ms,
                metadata=detail,
            )
        )


def _union_ms(intervals: list[tuple[float, float]]) -> float:
    if not intervals:
        return 0.0
    total = 0.0
    current_start, current_end = min(intervals, key=lambda item: item[0])
    for start, end in sorted(intervals, key=lambda item: item[0]):
        if start > current_end:
            total += current_end - current_start
            current_start, current_end = start, end
        elif end > current_end:
            current_end = end
    return total + (current_end - current_start)


def _merge_metadata(spans: list[StageSpan]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for span in spans:
        for key, value in span.metadata.items():
            previous = merged.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                merged[key] = (previous or 0) + value
            elif previous is None:
                merged[key] = value
    return merged
