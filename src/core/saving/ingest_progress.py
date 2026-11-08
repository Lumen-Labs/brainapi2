"""Durable source coverage and approximate ingestion forecasts."""

import logging
from datetime import datetime, timezone
from math import ceil, log as log_ratio
from time import monotonic
from uuid import uuid4

from src.core.saving import ingest_manifest

log = logging.getLogger(__name__)


class ChunkProgress:
    def __init__(self, brain_id, total_chars):
        from src.core.saving.pipeline_runtime import _context

        context = _context.get() or {}
        self.query = None
        self.started = monotonic()
        self.value = {
            "characters_processed": 0,
            "characters_total": total_chars,
            "chunks_completed": 0,
            "splits": 0,
        }
        if context.get("section_id") and context.get("brain_id") == brain_id:
            self.brain_id = brain_id
            self.query = {
                "_id": context["section_id"],
                "task_id": context["task_id"],
                "status": {"$nin": list(ingest_manifest.TERMINAL)},
            }
            self._write(token=uuid4().hex)

    def _write(self, *, token=None):
        if self.query is None:
            return
        now = datetime.now(timezone.utc).isoformat()
        changes = {
            "chunk_progress": {
                **self.value,
                "elapsed_seconds": round(monotonic() - self.started, 3),
                "updated_at": now,
            },
            "updated_at": now,
        }
        if token is not None:
            changes["chunk_progress_token"] = token
        try:
            result = ingest_manifest.sections_collection(self.brain_id).update_one(
                self.query, {"$set": changes}
            )
            if not result.matched_count:
                self.query = None
            elif token is not None:
                self.query["chunk_progress_token"] = token
        except Exception:
            if token is not None:
                self.query = None
            log.warning("Could not record ingestion chunk progress", exc_info=True)

    def complete(self, end):
        self.value["characters_processed"] = end
        self.value["chunks_completed"] += 1
        self._write()

    def split(self):
        self.value["splits"] += 1
        self._write()


def source_progress(rows):
    tracked = [row["chunk_progress"] for row in rows if row.get("chunk_progress") and row["status"] != "pending"]
    if not tracked:
        return None
    processed = 0
    for row in rows:
        if row["status"] == "completed":
            processed += row["chars"]
        elif row["status"] != "pending" and row.get("chunk_progress"):
            processed += min(row["chars"], row["chunk_progress"]["characters_processed"])
    return {
        "characters_processed": processed,
        "characters_total": sum(row["chars"] for row in rows),
        "chunks_completed": sum(progress["chunks_completed"] for progress in tracked),
        "splits": sum(progress["splits"] for progress in tracked),
    }


def publication_progress(run, section_task_ids):
    # The cursor counts all durable units, including discovery. Only sections
    # count here, and Explore remains gated until publication finishes.
    units = sorted(
        (unit["ordinal"], task)
        for task, unit in run["units"].items()
        if unit["state"] == "complete"
    )
    cursor = run.get("publication_cursor", 0)
    applied = sum(task in section_task_ids for _, task in units[:cursor])
    completed = sum(task in section_task_ids for _, task in units)
    phase = run["status"]
    visible = phase == "completed" or (phase == "cancelled" and run.get("mode") == "keep_saved")
    return {
        "staged_sections": completed - applied if phase in {"staging", "publishing"} else 0,
        "published_sections": applied if visible else 0,
        "applied_sections": applied if visible or phase == "publishing" else 0,
        "target_sections": completed if visible or phase in {"staging", "publishing"} else 0,
    }


def eta_range(rows, now):
    active = []
    rates = []
    queued = []
    for row in rows:
        status = row["status"]
        if status in {"pending", "queued"}:
            if row["chars"]:
                queued.append(row["chars"])
            continue
        if status not in {"started", "persisting", "consolidating", "completed"}:
            return None
        progress = row.get("chunk_progress")
        if not progress or not progress["characters_processed"] or not progress["elapsed_seconds"]:
            if status != "completed":
                return None
            continue
        rate = progress["elapsed_seconds"] / progress["characters_processed"]
        sample = (progress["characters_total"], rate)
        if status == "completed":
            rates.append(sample)
            continue
        remaining = progress["characters_total"] - progress["characters_processed"]
        if not remaining or status != "started":
            return None
        age = (now - datetime.fromisoformat(progress["updated_at"])).total_seconds()
        if progress["chunks_completed"] < 2 or age > max(
            180, 3 * progress["elapsed_seconds"] / progress["chunks_completed"]
        ):
            return None
        active.append((remaining * rate, progress["splits"]))
        rates.append(sample)

    if not active:
        return None
    # Active sections run concurrently. Queued work is bounded between using
    # the currently observed concurrency and running serially.
    queued_low = queued_high = 0
    for chars in queued:
        comparable = sorted(rates, key=lambda sample: abs(log_ratio(sample[0] / chars)))[:2]
        queued_low += chars * min(rate for _, rate in comparable) / len(active)
        queued_high += chars * max(rate for _, rate in comparable)
    lower = 0.65 * (max(seconds for seconds, _ in active) + queued_low)
    upper = max(seconds * (2 if splits else 1.6) for seconds, splits in active) + 2 * queued_high
    return {
        "phase": "staging",
        "min_seconds": max(1, ceil(lower)),
        "max_seconds": max(2, ceil(upper)),
        "estimated_at": now.isoformat(),
    }


def publishing_eta(run, now):
    cursor = run.get("publication_cursor", 0)
    remaining = sum(unit["state"] == "complete" for unit in run["units"].values()) - cursor
    started = run.get("publication_started_at")
    if cursor < 2 or remaining <= 0 or not started:
        return None
    elapsed = (now - datetime.fromisoformat(started)).total_seconds()
    estimate = max(1, elapsed / cursor * remaining)
    return {
        "phase": "publishing",
        "min_seconds": ceil(estimate * 0.5),
        "max_seconds": ceil(estimate * 2),
        "estimated_at": now.isoformat(),
    }
