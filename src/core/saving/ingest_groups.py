from __future__ import annotations

import copy
import re
from datetime import datetime, timezone
from typing import Any, Optional

GROUP_TRACKING_RE = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")
TASK_ID_CAP = 50
STAGE_NAMES = (
    "scout",
    "architect",
    "janitor",
    "observations",
    "consolidation",
    "embed",
    "index_rebuild",
)
STATUS_COUNTERS = {
    "completed": "completed",
    "failed": "failed",
    "partial_failed": "partial_failed",
}


class InvalidGroupTracking(ValueError):
    pass


def normalize_group_tracking(value: Any, *, strict: bool = True) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if GROUP_TRACKING_RE.fullmatch(text):
        return text
    if strict:
        raise InvalidGroupTracking(
            "group_tracking must be 1-128 characters matching [A-Za-z0-9._:-]"
        )
    return None


def empty_stage() -> dict[str, Any]:
    return {
        "input_tokens": 0,
        "output_tokens": 0,
        "cached_tokens": 0,
        "total_tokens": 0,
        "latency_ms": 0.0,
        "calls": 0,
    }


def empty_group_meter(group_tracking: str) -> dict[str, Any]:
    return {
        "group_tracking": group_tracking,
        "task_count": 0,
        "completed": 0,
        "failed": 0,
        "partial_failed": 0,
        "total_llm_tokens": 0,
        "total_input_tokens": 0,
        "total_output_tokens": 0,
        "total_cached_tokens": 0,
        "source_chars": 0,
        "source_tokens": 0,
        "stages": {name: empty_stage() for name in STAGE_NAMES},
        "task_ids": [],
        "updated_at": None,
    }


def normalize_meter_document(doc: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
    if not doc:
        return None
    key = str(doc.get("group_tracking") or doc.get("_id") or "").strip()
    if not key:
        return None
    meter = empty_group_meter(key)
    meter["task_count"] = int(doc.get("task_count") or 0)
    meter["completed"] = int(doc.get("completed") or 0)
    meter["failed"] = int(doc.get("failed") or 0)
    meter["partial_failed"] = int(doc.get("partial_failed") or 0)
    for field in (
        "total_llm_tokens",
        "total_input_tokens",
        "total_output_tokens",
        "total_cached_tokens",
        "source_chars",
        "source_tokens",
    ):
        meter[field] = int(doc.get(field) or 0)
    src_stages = doc.get("stages") or {}
    for name in STAGE_NAMES:
        src = src_stages.get(name) or {}
        dest = meter["stages"][name]
        dest["input_tokens"] = int(src.get("input_tokens") or 0)
        dest["output_tokens"] = int(src.get("output_tokens") or 0)
        dest["cached_tokens"] = int(src.get("cached_tokens") or 0)
        dest["total_tokens"] = int(src.get("total_tokens") or 0)
        dest["latency_ms"] = float(src.get("latency_ms") or 0.0)
        dest["calls"] = int(src.get("calls") or 0)
    ids = [str(item) for item in (doc.get("task_ids") or []) if item]
    meter["task_ids"] = ids[-TASK_ID_CAP:]
    updated_at = doc.get("updated_at")
    meter["updated_at"] = str(updated_at) if updated_at else None
    return meter


def apply_credit_to_meter(
    meter: dict[str, Any],
    *,
    task_id: str,
    status: str,
    cost: Optional[dict[str, Any]] = None,
    task_id_cap: int = TASK_ID_CAP,
) -> dict[str, Any]:
    updated = copy.deepcopy(meter) if meter else empty_group_meter("")
    updated["task_count"] = int(updated.get("task_count") or 0) + 1
    counter = STATUS_COUNTERS.get(status)
    if counter:
        updated[counter] = int(updated.get(counter) or 0) + 1
    payload = cost or {}
    for field in (
        "total_llm_tokens",
        "total_input_tokens",
        "total_output_tokens",
        "total_cached_tokens",
        "source_chars",
        "source_tokens",
    ):
        updated[field] = int(updated.get(field) or 0) + int(payload.get(field) or 0)
    stages = payload.get("stages") or {}
    dest_stages = updated.setdefault("stages", {})
    for name in STAGE_NAMES:
        src = stages.get(name) or {}
        dest = dest_stages.setdefault(name, empty_stage())
        dest["input_tokens"] = int(dest.get("input_tokens") or 0) + int(
            src.get("input_tokens") or 0
        )
        dest["output_tokens"] = int(dest.get("output_tokens") or 0) + int(
            src.get("output_tokens") or 0
        )
        dest["cached_tokens"] = int(dest.get("cached_tokens") or 0) + int(
            src.get("cached_tokens") or 0
        )
        dest["total_tokens"] = int(dest.get("total_tokens") or 0) + int(
            src.get("total_tokens") or 0
        )
        dest["latency_ms"] = float(dest.get("latency_ms") or 0.0) + float(
            src.get("latency_ms") or 0.0
        )
        dest["calls"] = int(dest.get("calls") or 0) + int(src.get("calls") or 0)
    ids = [item for item in (updated.get("task_ids") or []) if item != task_id]
    ids.append(task_id)
    updated["task_ids"] = ids[-int(task_id_cap) :]
    updated["updated_at"] = datetime.now(timezone.utc).isoformat()
    return updated


def credit_ingest_group(
    brain_id: str,
    group_tracking: Optional[str],
    task_id: str,
    status: str,
    cost: Optional[dict[str, Any]] = None,
) -> bool:
    key = normalize_group_tracking(group_tracking, strict=False)
    if not key or not task_id:
        return False
    from src.services.data.main import data_adapter

    return bool(
        data_adapter.credit_ingest_group(
            brain_id=brain_id,
            group_tracking=key,
            task_id=str(task_id),
            status=status,
            cost=cost or {},
        )
    )


def get_ingest_group(brain_id: str, group_tracking: str) -> Optional[dict[str, Any]]:
    key = normalize_group_tracking(group_tracking, strict=True)
    if not key:
        return None
    from src.services.data.main import data_adapter

    raw = data_adapter.get_ingest_group(brain_id=brain_id, group_tracking=key)
    return normalize_meter_document(raw) if raw else None
