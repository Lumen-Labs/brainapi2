import time
from typing import Any

from celery import signals

from src.lib.tracing.tracker import tracer

_task_starts: dict[str, float] = {}
_task_context_tokens: dict[str, Any] = {}


def _task_id(task_id: str | None, task=None) -> str:
    if task_id:
        return task_id
    request = getattr(task, "request", None)
    return getattr(request, "id", None) or "unknown"


def _task_name(task=None, sender=None) -> str:
    return getattr(task, "name", None) or getattr(sender, "name", None) or "unknown"


def _tenant_id(args=None, kwargs=None) -> str | None:
    args = args or ()
    kwargs = kwargs or {}
    candidates = list(args)
    candidates.extend(kwargs.values())
    for candidate in candidates:
        if isinstance(candidate, dict):
            brain_id = candidate.get("brain_id")
            if brain_id:
                return str(brain_id)
    brain_id = kwargs.get("brain_id")
    return str(brain_id) if brain_id else None


def _trace_id(task_id: str, kwargs=None) -> str:
    kwargs = kwargs or {}
    trace_id = kwargs.get("trace_id")
    return str(trace_id) if trace_id else task_id


def _metadata(task_id: str, task_name: str, args=None, kwargs=None) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "task_name": task_name,
        "args_count": len(args or ()),
        "kwargs_keys": sorted((kwargs or {}).keys()),
    }


def install_celery_tracing(_app=None, service_name: str = "brainapi-worker") -> None:
    signals.task_prerun.connect(
        _trace_task_prerun,
        weak=False,
        dispatch_uid=f"{service_name}.trace.task_prerun",
    )
    signals.task_postrun.connect(
        _trace_task_postrun,
        weak=False,
        dispatch_uid=f"{service_name}.trace.task_postrun",
    )
    signals.task_failure.connect(
        _trace_task_failure,
        weak=False,
        dispatch_uid=f"{service_name}.trace.task_failure",
    )


def _trace_task_prerun(sender=None, task_id=None, task=None, args=None, kwargs=None, **_):
    resolved_task_id = _task_id(task_id, task)
    resolved_task_name = _task_name(task, sender)
    _task_starts[resolved_task_id] = time.perf_counter()
    _task_context_tokens[resolved_task_id] = tracer.set_context(
        trace_id=_trace_id(resolved_task_id, kwargs),
        tenant_id=_tenant_id(args, kwargs),
    )
    tracer.publish(
        "latency",
        f"{resolved_task_name}.started",
        service="brainapi-worker",
        operation=resolved_task_name,
        metadata=_metadata(resolved_task_id, resolved_task_name, args, kwargs),
    )


def _trace_task_postrun(sender=None, task_id=None, task=None, args=None, kwargs=None, retval=None, state=None, **_):
    resolved_task_id = _task_id(task_id, task)
    resolved_task_name = _task_name(task, sender)
    started_at = _task_starts.pop(resolved_task_id, None)
    duration_ms = (time.perf_counter() - started_at) * 1000 if started_at else None
    metadata = {
        **_metadata(resolved_task_id, resolved_task_name, args, kwargs),
        "state": state,
        "returned": retval is not None,
    }
    if duration_ms is not None and duration_ms >= tracer.default_sla_ms:
        tracer.publish(
            "sla_breach",
            f"{resolved_task_name}.sla_breach",
            service="brainapi-worker",
            operation=resolved_task_name,
            duration_ms=duration_ms,
            threshold_ms=tracer.default_sla_ms,
            metadata=metadata,
        )
    token = _task_context_tokens.pop(resolved_task_id, None)
    if token is not None:
        tracer.reset_context(token)


def _trace_task_failure(sender=None, task_id=None, exception=None, args=None, kwargs=None, einfo=None, **_):
    resolved_task_id = _task_id(task_id, sender)
    resolved_task_name = _task_name(sender=sender)
    started_at = _task_starts.get(resolved_task_id)
    duration_ms = (time.perf_counter() - started_at) * 1000 if started_at else None
    metadata = _metadata(resolved_task_id, resolved_task_name, args, kwargs)
    if einfo is not None:
        metadata["traceback"] = str(einfo)
    tracer.exception(
        f"{resolved_task_name}.failed",
        exception or RuntimeError("Celery task failed"),
        service="brainapi-worker",
        operation=resolved_task_name,
        duration_ms=duration_ms,
        metadata=metadata,
    )
