import os
import time
from typing import Any, Callable, Optional
from uuid import uuid4

import httpx

RUNTIME_EVENT = "brainapi_runtime_operation_completed"
INGESTION_EVENT = "ingestion_completed"
RUNTIME_DISTINCT_ID = "brainapi-worker"

INGEST_DATA_OPERATION = "ingest_data_task"
INGEST_STRUCTURED_OPERATION = "ingest_structured_task"
INGEST_FILE_OPERATION = "ingest_file_task"
INGEST_URL_OPERATION = "ingest_url_task"

KINDS = ("text", "json", "file", "structured", "url")
PIPELINE_MODES = ("lightweight", "accurate")
LLM_STAGES = (
    "scout",
    "architect",
    "janitor",
    "observations",
    "consolidation",
)
RUNTIME_OUTCOMES = {
    "completed": "succeeded",
    "failed": "failed",
    "partial_failed": "partial",
}
INGESTION_OUTCOMES = {
    "completed": "done",
    "failed": "failed",
    "partial_failed": "partial",
}
PROPERTY_KEYS = (
    "duration_ms",
    "kind",
    "pipeline_mode",
    "size_chars",
    "size_pages",
    "llm_call_count",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "unit_count",
    "outcome",
    "operation",
)

_captured: list[dict[str, Any]] = []
_STATE: dict[str, Any] = {"transport": None, "retain": False}


def now_ms() -> int:
    return max(0, int(time.time() * 1000))


def duration_ms(started_at_ms: Any) -> int:
    if started_at_ms is None:
        return 0
    try:
        started = int(started_at_ms)
    except (TypeError, ValueError):
        return 0
    return max(0, now_ms() - started)


def captured_events() -> list[dict[str, Any]]:
    return list(_captured)


def reset_captured_events() -> None:
    _captured.clear()


def enable_event_capture() -> None:
    _STATE["retain"] = True


def disable_event_capture() -> None:
    _STATE["retain"] = False
    _captured.clear()


def set_capture_transport(transport: Optional[Callable[[dict[str, Any]], None]]) -> None:
    _STATE["transport"] = transport


def kind_for_ingest_data(payload) -> str:
    explicit = getattr(payload, "kind", None)
    if explicit in KINDS:
        return explicit
    data = getattr(payload, "data", None)
    data_type = getattr(data, "data_type", None)
    if data_type in ("text", "json"):
        return data_type
    return "text"


def telemetry_extra(
    *,
    operation: str,
    kind: str,
    pipeline_mode: str,
    **metrics: Any,
) -> dict[str, Any]:
    extra = {
        "telemetry_operation": operation,
        "kind": kind if kind in KINDS else "text",
        "pipeline_mode": (
            pipeline_mode if pipeline_mode in PIPELINE_MODES else "accurate"
        ),
    }
    started_at_ms = metrics.get("started_at_ms")
    size_chars = metrics.get("size_chars")
    size_pages = metrics.get("size_pages")
    unit_count = metrics.get("unit_count")
    if started_at_ms is not None:
        extra["started_at_ms"] = max(0, int(started_at_ms))
    if size_chars is not None:
        extra["size_chars"] = max(0, int(size_chars))
    if size_pages is not None:
        extra["size_pages"] = max(0, int(size_pages))
    if unit_count is not None:
        extra["unit_count"] = max(0, int(unit_count))
    return extra


def _int_field(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _llm_usage(cost: dict[str, Any]) -> tuple[int, int, int]:
    stages = cost.get("stages") or {}
    prompt = completion = calls = 0
    for name in LLM_STAGES:
        stage = stages.get(name) or {}
        prompt += _int_field(stage.get("input_tokens"))
        completion += _int_field(stage.get("output_tokens"))
        calls += _int_field(stage.get("calls"))
    return calls, prompt, completion


def _unit_count(payload: dict[str, Any], cost: dict[str, Any], counts: dict[str, Any]) -> int:
    if payload.get("unit_count") is not None:
        return _int_field(payload.get("unit_count"))
    for value in (
        cost.get("architect_units"),
        counts.get("relationships"),
        counts.get("triples"),
        payload.get("size_pages"),
        counts.get("pages"),
    ):
        if value is not None:
            return _int_field(value)
    return 0


def properties_from_status(payload: dict[str, Any]) -> dict[str, Any]:
    cost = payload.get("cost") if isinstance(payload.get("cost"), dict) else {}
    counts = payload.get("counts") if isinstance(payload.get("counts"), dict) else {}
    calls, prompt, completion = _llm_usage(cost)
    kind = payload.get("kind") if payload.get("kind") in KINDS else "text"
    pipeline_mode = payload.get("pipeline_mode")
    if pipeline_mode not in PIPELINE_MODES:
        pipeline_mode = "accurate"
    size_chars = payload.get("size_chars")
    if size_chars is None:
        size_chars = cost.get("source_chars")
    size_pages = payload.get("size_pages")
    if size_pages is None:
        size_pages = counts.get("pages")
    return {
        "duration_ms": duration_ms(payload.get("started_at_ms")),
        "kind": kind,
        "pipeline_mode": pipeline_mode,
        "size_chars": _int_field(size_chars),
        "size_pages": _int_field(size_pages),
        "llm_call_count": calls,
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
        "unit_count": _unit_count(payload, cost, counts),
    }


def stamp_started_at(payload: dict[str, Any], status: str) -> None:
    if status == "started" and payload.get("started_at_ms") is None:
        payload["started_at_ms"] = now_ms()


def maybe_emit_ingestion_telemetry(previous: Optional[str], payload: dict[str, Any]) -> None:
    status = payload.get("status")
    if previous in RUNTIME_OUTCOMES or status not in RUNTIME_OUTCOMES:
        return
    operation = payload.get("telemetry_operation")
    if not operation:
        return
    metrics = properties_from_status(payload)
    runtime_props = {
        **metrics,
        "operation": operation,
        "outcome": RUNTIME_OUTCOMES[status],
    }
    ingestion_props = {
        **metrics,
        "outcome": INGESTION_OUTCOMES[status],
    }
    capture_event(RUNTIME_EVENT, runtime_props)
    capture_event(INGESTION_EVENT, ingestion_props)


def capture_event(event: str, properties: dict[str, Any]) -> None:
    safe = {key: properties[key] for key in PROPERTY_KEYS if key in properties}
    record = {
        "event": event,
        "distinct_id": RUNTIME_DISTINCT_ID,
        "properties": safe,
    }
    if _STATE.get("retain"):
        _captured.append(record)
    try:
        transport = _STATE.get("transport")
        if transport is not None:
            transport(record)
            return
        _send_posthog(record)
    except Exception:
        return


def _send_posthog(record: dict[str, Any]) -> None:
    token = os.getenv("POSTHOG_PROJECT_TOKEN")
    if not token or os.getenv("POSTHOG_ENABLED") == "false":
        return
    host = (os.getenv("POSTHOG_HOST") or "https://us.i.posthog.com").rstrip("/")
    httpx.post(
        f"{host}/batch/",
        json={
            "api_key": token,
            "batch": [
                {
                    "uuid": str(uuid4()),
                    "event": record["event"],
                    "distinct_id": record["distinct_id"],
                    "properties": {
                        **record["properties"],
                        "distinct_id": record["distinct_id"],
                        "$process_person_profile": False,
                        "$geoip_disable": True,
                    },
                }
            ],
        },
        timeout=1.5,
    )
