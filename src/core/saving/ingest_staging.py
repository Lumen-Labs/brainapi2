"""State transitions for imports whose workers write only to staging."""

import logging
from time import time
from uuid import NAMESPACE_URL, uuid4, uuid5

from src.core.saving import ingest_cancellation

CLAIM_SECONDS = 120
DISPATCH_RETRY_SECONDS = 30
MAX_UNIT_FAILURES = 3
MAX_SETTLEMENT_FAILURES = 5
MAX_ROLLBACK_FAILURES = MAX_SETTLEMENT_FAILURES
REMOTE_OCR_DEADLINE_SECONDS = 3600
SMALL_REMOTE_OCR_DEADLINE_SECONDS = 300
SMALL_REMOTE_OCR_MAX_BYTES = 1024 * 1024
OCR_PAGE_IDLE_SECONDS = 120
REMOTE_OCR_TIMEOUT_ERROR = "Document parser did not return a result before the deadline"
REMOTE_OCR_REJECTED_ERROR = "Document parser rejected the ingestion request"
REMOTE_OCR_EMPTY_ERROR = "Ingestion produced no text"
TERMINAL_TASK_STATUSES = {"completed", "failed", "partial_failed"}
_SOFT_PAGE_FAILURE_MARKERS = (
    "ingestion produced no text",
    "success callback rejected",
    "success callback failed",
)
log = logging.getLogger(__name__)


def create_run(
    brain_id, root_id, *, callback_deadline=None, upload_deadline=None,
    message=None, discovery=False, small_remote_ocr=False,
):
    """Create the run and, for direct submissions, its first dispatch intent."""
    unit_id = None
    units = {}
    if message is not None:
        key = "discovery" if discovery else "source"
        unit_id = uuid5(NAMESPACE_URL, f"brainapi:ingest:{root_id}:{key}").hex
        units[unit_id] = {"state": "pending", "ordinal": 0, "message": message}
    inserted = {
        "status": "staging",
        "staged": True,
        "revision": "",
        "units": units,
        "publication_cursor": 0,
        "sealed": message is not None and not discovery,
        "callback_deadline": callback_deadline,
        "upload_deadline": upload_deadline,
    }
    if small_remote_ocr:
        inserted["small_remote_ocr"] = True
    ingest_cancellation._runs(brain_id).update_one(
        {"_id": root_id},
        {"$setOnInsert": inserted},
        upsert=True,
    )
    return unit_id


def expire_file_upload(brain_id, root_id):
    def transition(run):
        deadline = run.get("upload_deadline")
        if run["status"] == "staging" and not run["units"] and deadline and deadline <= time():
            return {"status": "discarding", "discard_result": "failed"}
        return None

    return ingest_cancellation._change(brain_id, root_id, transition)


def register_unit(brain_id, root_id, key, message, *, seal=False, task_id=None):
    """Persist a dispatch intent before sending its broker message."""
    unit_id = task_id or uuid5(NAMESPACE_URL, f"brainapi:ingest:{root_id}:{key}").hex

    def transition(run):
        if run["status"] != "staging" or run.get("sealed") or unit_id in run["units"]:
            return None
        return {"sealed": seal, "units": {
            **run["units"],
            unit_id: {
                "state": "pending",
                "ordinal": len(run["units"]),
                "message": message,
            },
        }}

    run = ingest_cancellation._change(brain_id, root_id, transition)
    if run and run["status"] == "staging" and unit_id in run["units"]:
        return unit_id
    return None


def ocr_unit_id(root_id, page_index):
    return uuid5(NAMESPACE_URL, f"brainapi:ingest:{root_id}:ocr:{page_index}").hex


def _ocr_counts(run):
    pages = (run or {}).get("ocr_pages") or {}
    accepted = sum(1 for state in pages.values() if state == "accepted")
    failed = sum(1 for state in pages.values() if state == "failed")
    return accepted, failed


def _ocr_page_key(page_index, pages):
    if page_index is None:
        return f"anon:{len(pages)}"
    return str(int(page_index))


def _ocr_touch(run, total_pages=None):
    now = time()
    changes = {
        "ocr_last_callback_at": now,
        "ocr_idle_seal_after": now + OCR_PAGE_IDLE_SECONDS,
    }
    stored = int(run.get("ocr_total_pages") or 0)
    if total_pages:
        incoming = int(total_pages)
        if incoming > stored:
            changes["ocr_total_pages"] = incoming
    if run.get("small_remote_ocr") and not (run.get("ocr_pages") or {}):
        changes["callback_deadline"] = now + SMALL_REMOTE_OCR_DEADLINE_SECONDS
    return now, changes


def _ocr_incomplete(run, accepted, failed):
    total = run.get("ocr_total_pages")
    if failed:
        return True
    return bool(total) and (accepted + failed) < int(total)


def _ocr_close_action(run, now, *, rejected=False):
    accepted, failed = _ocr_counts(run)
    accounted = accepted + failed
    total = run.get("ocr_total_pages")
    past_deadline = run.get("callback_deadline") is not None and run["callback_deadline"] <= now
    finished = (total and accounted >= int(total)) or rejected or past_deadline
    if not finished:
        if total:
            return None
        idle_after = run.get("ocr_idle_seal_after")
        if idle_after is None or idle_after > now or not accounted:
            return None
    if not accepted:
        return "fail"
    if _ocr_incomplete(run, accepted, failed):
        return "partial"
    return "seal"


def _ocr_partial_error(accepted, failed, total):
    total = int(total or (accepted + failed))
    missing = max(0, total - accepted - failed)
    return (
        f"OCR finished with {accepted} of {total} pages accepted; "
        f"{failed} failed, {missing} missing"
    )


def _apply_ocr_close(changes, run, now, *, rejected=False):
    merged = {**run, **changes}
    if "ocr_pages" in changes:
        merged["ocr_pages"] = changes["ocr_pages"]
    action = _ocr_close_action(merged, now, rejected=rejected)
    if action in {"seal", "partial"} and not merged.get("sealed"):
        changes["sealed"] = True
        accepted, failed = _ocr_counts(merged)
        changes["ocr_pages_accepted"] = accepted
        changes["ocr_pages_failed"] = failed
        if action == "partial":
            changes["ocr_partial"] = True
            changes["error"] = _ocr_partial_error(
                accepted, failed, merged.get("ocr_total_pages"),
            )
    elif action == "fail" and merged.get("status") == "staging":
        changes["status"] = "failed"
        changes["error"] = merged.get("last_ocr_error") or REMOTE_OCR_TIMEOUT_ERROR
    return changes


def register_ocr_callback(brain_id, root_id, message, page_index=None, total_pages=None):
    """Accept one page of OCR text. Replays of the same page_index stay idempotent."""
    outcome = {"unit_id": None, "created": False, "rejected": True}

    def transition(run):
        deadline = run.get("callback_deadline")
        pages = dict(run.get("ocr_pages") or {})
        closed = (
            run["status"] != "staging"
            or run.get("sealed")
            or (deadline is not None and deadline <= time())
        )
        if page_index is not None:
            unit_id = ocr_unit_id(root_id, int(page_index))
            key = _ocr_page_key(page_index, pages)
            if unit_id in run.get("units", {}) or pages.get(key) == "accepted":
                outcome.update(unit_id=unit_id, created=False, rejected=closed)
                return None
            if closed:
                return None
        else:
            if closed:
                return None
            accepted, failed = _ocr_counts(run)
            unit_id = ocr_unit_id(root_id, accepted + failed)
            key = _ocr_page_key(None, pages)
            while unit_id in run.get("units", {}):
                accepted += 1
                unit_id = ocr_unit_id(root_id, accepted + failed)
        unit = {
            "state": "pending",
            "ordinal": len(run["units"]),
            "message": message,
        }
        pages[key] = "accepted"
        now, changes = _ocr_touch(run, total_pages)
        changes["units"] = {**run["units"], unit_id: unit}
        changes["ocr_pages"] = pages
        accepted, failed = _ocr_counts({**run, "ocr_pages": pages})
        changes["ocr_pages_accepted"] = accepted
        changes["ocr_pages_failed"] = failed
        changes = _apply_ocr_close(changes, run, now)
        outcome.update(unit_id=unit_id, created=True, rejected=False)
        return changes

    run = ingest_cancellation._change(brain_id, root_id, transition)
    if outcome["rejected"] or not outcome["unit_id"]:
        return None
    if outcome["created"]:
        _sync_ocr_task_status(brain_id, root_id, run)
        maybe_finish_remote_ocr(brain_id, root_id)
    return outcome["unit_id"], outcome["created"]


def _ocr_status_extra(run):
    accepted, failed = _ocr_counts(run)
    if not accepted:
        accepted = _accepted_ocr_count(run, run)
    if not failed:
        try:
            failed = int(run.get("ocr_pages_failed") or 0)
        except (TypeError, ValueError):
            failed = 0
    extra = {
        "ocr_pages_failed": failed,
        "remote_ocr_deadline": run.get("callback_deadline") or run.get("remote_ocr_deadline"),
    }
    if accepted:
        extra["ocr_pages_accepted"] = accepted
    total = run.get("ocr_total_pages")
    if total:
        extra["ocr_total_pages"] = total
    if run.get("ocr_partial"):
        extra["ocr_partial"] = True
    return extra, accepted, failed


def _sync_ocr_task_status(brain_id, task_id, run):
    if not run:
        return None
    extra, accepted, failed = _ocr_status_extra(run)
    try:
        from src.workers.tasks.ingestion import set_ingestion_task_status

        if run.get("ocr_partial"):
            return set_ingestion_task_status(
                task_id,
                brain_id,
                "partial_failed",
                stage="partial_failed",
                error=_ocr_partial_error(accepted, failed, run.get("ocr_total_pages")),
                extra=extra,
                counts={
                    "ocr_pages_accepted": accepted,
                    "ocr_pages_failed": failed,
                    "ocr_total_pages": run.get("ocr_total_pages"),
                },
            )
        return set_ingestion_task_status(
            task_id,
            brain_id,
            "started",
            stage="processing",
            extra=extra,
        )
    except Exception:
        log.debug("Could not record OCR status for %s", task_id, exc_info=True)
        return None


def _update_unit(brain_id, root_id, unit_id, update):
    def transition(run):
        unit = run.get("units", {}).get(unit_id)
        if run["status"] != "staging" or unit is None:
            return None
        replacement = update(unit)
        if replacement is None:
            return None
        return {"units": {**run["units"], unit_id: replacement}}

    run = ingest_cancellation._change(brain_id, root_id, transition)
    if run and run["status"] == "staging":
        return run["units"].get(unit_id)
    return None


def claim_unit(brain_id, root_id, unit_id):
    token = uuid4().hex

    def claim(unit):
        now = time()
        if unit["state"] == "complete" or unit.get("expires_at", 0) > now:
            return None
        return {**unit, "state": "running", "token": token, "expires_at": now + CLAIM_SECONDS}

    unit = _update_unit(brain_id, root_id, unit_id, claim)
    return token if unit and unit.get("token") == token else None


def renew_claim(brain_id, root_id, unit_id, token):
    def renew(unit):
        now = time()
        if unit.get("token") != token or unit["expires_at"] <= now:
            return None
        return {**unit, "expires_at": now + CLAIM_SECONDS}

    unit = _update_unit(brain_id, root_id, unit_id, renew)
    return bool(unit and unit.get("token") == token and unit["expires_at"] > time())


def complete_unit(brain_id, root_id, unit_id, token, output_ref):
    """Commit a reference to immutable, attempt-scoped staged output."""
    def complete(unit):
        if unit.get("token") != token or unit["expires_at"] <= time():
            return None
        return {
            "state": "complete",
            "ordinal": unit["ordinal"],
            "attempt": token,
            "output_ref": output_ref,
        }

    unit = _update_unit(brain_id, root_id, unit_id, complete)
    if not (unit and unit.get("attempt") == token and unit.get("output_ref") == output_ref):
        return False
    close_staging(brain_id, root_id)
    return True


def record_unit_failure(brain_id, root_id, unit_id, token):
    """Retry failed work with backoff, then discard an exhausted run."""
    def transition(run):
        unit = run.get("units", {}).get(unit_id, {})
        now = time()
        if (
            run["status"] != "staging"
            or unit.get("token") != token
            or unit["expires_at"] <= now
        ):
            return None
        failures = unit.get("failures", 0) + 1
        if failures >= MAX_UNIT_FAILURES:
            return {"status": "discarding", "discard_result": "failed"}
        retry = {
            **unit,
            "state": "pending",
            "failures": failures,
            "expires_at": 0,
            "dispatch_after": now + DISPATCH_RETRY_SECONDS,
        }
        return {"units": {**run["units"], unit_id: retry}}

    return ingest_cancellation._change(brain_id, root_id, transition)


def reserve_dispatch(brain_id, root_id, unit_id):
    """Return the persisted message once per dispatch retry window."""
    dispatch_id = uuid4().hex

    def reserve(unit):
        now = time()
        if (
            unit["state"] == "complete"
            or unit.get("expires_at", 0) > now
            or unit.get("dispatch_after", 0) > now
        ):
            return None
        return {
            **unit,
            "dispatch_id": dispatch_id,
            "dispatch_after": now + DISPATCH_RETRY_SECONDS,
        }

    unit = _update_unit(brain_id, root_id, unit_id, reserve)
    if unit and unit.get("dispatch_id") == dispatch_id:
        return unit["message"]
    return None


def pending_dispatches(brain_id, root_id):
    run = ingest_cancellation._runs(brain_id).find_one({"_id": root_id})
    if not run or run["status"] != "staging":
        return []
    now = time()
    return [
        (unit_id, unit["message"])
        for unit_id, unit in run["units"].items()
        if unit["state"] != "complete"
        and unit.get("expires_at", 0) <= now
        and unit.get("dispatch_after", 0) <= now
    ]


def close_staging(brain_id, root_id, mode=None):
    """Freeze the set of completed units before publishing or discarding."""
    if mode not in (None, "keep_saved", "rollback"):
        raise ValueError("Unknown cancellation mode")

    def transition(run):
        if run["status"] != "staging":
            return None
        if mode is None and (
            not run.get("sealed")
            or (not run["units"] and "discovered" not in run)
            or any(unit["state"] != "complete" for unit in run["units"].values())
        ):
            return None
        return {"status": "discarding" if mode == "rollback" else "publishing", "mode": mode}

    return ingest_cancellation._change(brain_id, root_id, transition)


def seal_units(brain_id, root_id):
    """Close discovery after every dispatch intent has been registered."""
    return ingest_cancellation._change(
        brain_id, root_id,
        lambda run: {"sealed": True}
        if run["status"] == "staging" and not run.get("sealed") else None,
    )


def _completed_units(run):
    units = (run or {}).get("units") or {}
    if not isinstance(units, dict):
        return []
    return sorted(
        (
            unit
            for unit in units.values()
            if isinstance(unit, dict) and unit.get("state") == "complete"
        ),
        key=lambda unit: unit.get("ordinal", 0),
    )


def publication_operation_id(root_id, cursor):
    return uuid5(NAMESPACE_URL, f"brainapi:publish:{root_id}:{cursor}").hex


def next_publication_unit(brain_id, root_id):
    """Read the durable cursor. Replaying the same position uses the same ID."""
    run = ingest_cancellation._runs(brain_id).find_one({"_id": root_id})
    if not run or run["status"] != "publishing":
        return None
    cursor = run["publication_cursor"]
    units = _completed_units(run)
    if cursor >= len(units):
        return None
    return {
        "cursor": cursor,
        "operation_id": publication_operation_id(root_id, cursor),
        "output_ref": units[cursor]["output_ref"],
    }


def advance_publication(brain_id, root_id, cursor):
    """Advance after the adapter confirms the operation is durable."""
    def transition(run):
        if run["status"] != "publishing" or run["publication_cursor"] != cursor:
            return None
        if cursor >= len(_completed_units(run)):
            return None
        return {"publication_cursor": cursor + 1}

    return ingest_cancellation._change(brain_id, root_id, transition)


def published_units_before_cursor(brain_id, root_id):
    """Completed units already applied before the current publication cursor."""
    run = ingest_cancellation._runs(brain_id).find_one({"_id": root_id})
    if not run or run.get("status") != "publishing":
        return []
    units = _completed_units(run)
    cursor = int(run.get("publication_cursor") or 0)
    return units[: max(0, min(len(units), cursor))]


def units_to_unpublish(brain_id, root_id):
    """Units that reached live stores, including the partially applied cursor unit."""
    run = ingest_cancellation._runs(brain_id).find_one({"_id": root_id})
    if not run or run["status"] not in {"publish_partial", "rollback_failed"}:
        return []
    units = _completed_units(run)
    if not units:
        return []
    cursor = run.get("publication_cursor", 0)
    return units[: min(len(units), cursor + 1)]


def _terminal_units(run):
    return {
        unit_id: {"state": unit["state"], "ordinal": unit["ordinal"]}
        for unit_id, unit in run["units"].items()
    }


def finish_publication(brain_id, root_id):
    """Record completion once every selected unit has been applied."""
    def transition(run):
        if run["status"] != "publishing":
            return None
        if run["publication_cursor"] != len(_completed_units(run)):
            return None
        terminal = "cancelled" if run.get("mode") else (
            "partial_failed" if run.get("ocr_partial") else "completed"
        )
        return {
            "status": terminal,
            "units": _terminal_units(run),
            "pending_cleanup": True,
        }

    return ingest_cancellation._change(brain_id, root_id, transition)


def finish_discard(brain_id, root_id):
    """Record cancellation after staged output cleanup is durable."""
    return ingest_cancellation._change(
        brain_id, root_id,
        lambda run: {
            "status": run.get("discard_result", "cancelled"),
            "units": _terminal_units(run),
        } if run["status"] == "discarding" else None,
    )


def _terminal_settlement_status(run):
    if run["status"] == "publishing":
        if run.get("publication_cursor", 0) >= len(_completed_units(run)):
            return "cancelled" if run.get("mode") else (
                "partial_failed" if run.get("ocr_partial") else "completed"
            )
        return "publish_partial"
    if run["status"] == "publish_partial":
        return "rollback_failed"
    if run["status"] == "discarding" and run.get("mode") == "rollback":
        return "rollback_failed"
    if run["status"] == "discarding":
        return run.get("discard_result", "failed")
    return "failed"


def record_settlement_failure(brain_id, root_id, phase):
    """Bound discard and publish retries, then fail without holding the brain gate."""
    def transition(run):
        if run["status"] != phase or phase not in {"discarding", "publishing", "publish_partial"}:
            return None
        failures = run.get("settlement_failures", 0) + 1
        changes = {"settlement_failures": failures}
        if failures >= MAX_SETTLEMENT_FAILURES:
            status = _terminal_settlement_status(run)
            changes["status"] = status
            if status == "publish_partial":
                changes["settlement_failures"] = 0
            elif status == "rollback_failed" and run["status"] == "publish_partial":
                changes["pending_cleanup"] = False
                if run.get("rollback_reason"):
                    changes["rollback_reason"] = run["rollback_reason"]
            else:
                changes.update(units=_terminal_units(run), pending_cleanup=True)
        return changes

    run = ingest_cancellation._change(brain_id, root_id, transition)
    return bool(run and run["status"] != phase)


def finish_publish_rollback(brain_id, root_id):
    """Mark a partial publish failed only after live writes have been undone."""
    return ingest_cancellation._change(
        brain_id, root_id,
        lambda run: {
            "status": "failed",
            "units": _terminal_units(run),
            "pending_cleanup": True,
        } if run["status"] in {"publish_partial", "rollback_failed"} else None,
    )


def graph_touch(brain_id, root_id, operation_id):
    run = ingest_cancellation._runs(brain_id).find_one({"_id": root_id})
    if not run:
        return None
    return (run.get("graph_touches") or {}).get(operation_id)


def graph_touches(brain_id, root_id):
    run = ingest_cancellation._runs(brain_id).find_one({"_id": root_id})
    if not run:
        return {}
    return dict(run.get("graph_touches") or {})


def record_graph_touch(brain_id, root_id, operation_id, touches):
    """Persist created/reused markers before a graph MERGE is applied."""
    def transition(run):
        if run["status"] not in {"publishing", "publish_partial"}:
            return None
        ledger = dict(run.get("graph_touches") or {})
        if operation_id in ledger:
            return None
        ledger[operation_id] = touches
        return {"graph_touches": ledger}

    return ingest_cancellation._change(brain_id, root_id, transition)


def rollback_done_ids(brain_id, root_id):
    run = ingest_cancellation._runs(brain_id).find_one({"_id": root_id})
    if not run:
        return []
    return list(run.get("rollback_done") or [])


def mark_rollback_done(brain_id, root_id, item_ids):
    def transition(run):
        if run["status"] not in {"publish_partial", "rollback_failed"}:
            return None
        done = list(run.get("rollback_done") or [])
        known = set(done)
        added = False
        for item_id in item_ids:
            if item_id in known:
                continue
            done.append(item_id)
            known.add(item_id)
            added = True
        if not added:
            return None
        return {"rollback_done": done}

    return ingest_cancellation._change(brain_id, root_id, transition)


def set_rollback_reason(brain_id, root_id, reason):
    def transition(run):
        if run["status"] not in {"publish_partial", "rollback_failed"}:
            return None
        if run.get("rollback_reason") == reason:
            return None
        return {"rollback_reason": reason}

    return ingest_cancellation._change(brain_id, root_id, transition)


def abort_publish_rollback(brain_id, root_id, reason):
    """Fail closed without mutating live data when undo cannot be completed safely."""
    def transition(run):
        if run["status"] not in {"publish_partial", "rollback_failed"}:
            return None
        if (
            run["status"] == "rollback_failed"
            and run.get("rollback_reason") == reason
            and run.get("pending_cleanup") is False
        ):
            return None
        return {
            "status": "rollback_failed",
            "pending_cleanup": False,
            "rollback_reason": reason,
        }

    return ingest_cancellation._change(brain_id, root_id, transition)


def record_rollback_failure(brain_id, root_id):
    """Report exhausted cleanup retries without holding the brain gate indefinitely."""
    return record_settlement_failure(brain_id, root_id, "discarding")


def needs_abandoned_cleanup(run):
    return bool(run.get("staged") and run.get("pending_cleanup") is True)


def mark_cleanup_complete(brain_id, root_id):
    return ingest_cancellation._change(
        brain_id, root_id,
        lambda run: {"pending_cleanup": False} if run.get("pending_cleanup") is not False else None,
    )


def complete_discovery(brain_id, root_id, unit_id, token, messages, *, wait_for_sections=False):
    """Atomically register discovered source units and seal the run."""
    def transition(run):
        if run["status"] != "staging" or run.get("sealed"):
            return None
        units = run["units"]
        discovery = units.get(unit_id, {})
        if discovery.get("token") != token or discovery["expires_at"] <= time():
            return None
        children = {
            uuid5(NAMESPACE_URL, f"brainapi:ingest:{root_id}:page:{index}").hex: {
                "state": "pending",
                "ordinal": index,
                "message": message,
            }
            for index, message in enumerate(messages)
        }
        return {"units": children, "sealed": not wait_for_sections, "discovered": len(children)}

    return ingest_cancellation._change(brain_id, root_id, transition)


def is_remote_ocr_run(brain_id, root_id):
    run = ingest_cancellation._runs(brain_id).find_one({"_id": root_id})
    return bool(run and run.get("callback_deadline"))


def is_small_remote_ocr(size_bytes):
    return size_bytes is not None and size_bytes <= SMALL_REMOTE_OCR_MAX_BYTES


def remote_ocr_callback_deadline(size_bytes=None):
    return time() + REMOTE_OCR_DEADLINE_SECONDS


def is_soft_ocr_page_failure(error):
    lowered = str(error or "").lower()
    return any(marker in lowered for marker in _SOFT_PAGE_FAILURE_MARKERS)


def has_accepted_ocr_pages(run=None, cached=None):
    cached = cached or {}
    if cached.get("ocr_pages_accepted"):
        return True
    run = run or {}
    if int(run.get("ocr_pages_accepted") or 0) > 0:
        return True
    pages = run.get("ocr_pages") or {}
    return any(state == "accepted" for state in pages.values())


def _mark_remote_ocr_failed(brain_id, task_id, error):
    try:
        from src.workers.tasks.ingestion import set_ingestion_task_status

        return set_ingestion_task_status(
            task_id,
            brain_id,
            "failed",
            stage="failed",
            error=error,
            extra={"remote_ocr_deadline": None},
        )
    except Exception:
        log.exception("Could not record failed ingest status for %s", task_id)
        return None


def _accepted_ocr_count(run, cached):
    run = run or {}
    cached = cached or {}
    accepted, _failed = _ocr_counts(run)
    if accepted:
        return accepted
    value = cached.get("ocr_pages_accepted")
    if value is True:
        return 1
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _should_report_ocr_partial(run, cached):
    run = run or {}
    cached = cached or {}
    if run.get("ocr_partial") or cached.get("ocr_partial"):
        return True
    accepted = _accepted_ocr_count(run, cached)
    failed = int(run.get("ocr_pages_failed") or 0) or sum(
        1 for state in (run.get("ocr_pages") or {}).values() if state == "failed"
    )
    if not failed:
        failed = int(cached.get("ocr_pages_failed") or 0)
    total = run.get("ocr_total_pages") or cached.get("ocr_total_pages")
    deadline = run.get("callback_deadline")
    if deadline is None:
        deadline = cached.get("remote_ocr_deadline")
    closed = bool(run.get("sealed")) or (deadline is not None and deadline <= time())
    if not accepted or not closed:
        return False
    return _ocr_incomplete({**cached, **run, "ocr_total_pages": total}, accepted, failed)


def expire_remote_ocr(brain_id, root_id, *, rejected=False):
    default_error = REMOTE_OCR_REJECTED_ERROR if rejected else REMOTE_OCR_TIMEOUT_ERROR

    def transition(run):
        deadline = run.get("callback_deadline")
        if run["status"] != "staging" or not deadline:
            return None
        now = time()
        changes = _apply_ocr_close({}, run, now, rejected=rejected)
        if not changes:
            return None
        if changes.get("status") == "failed" and not changes.get("error"):
            changes["error"] = run.get("last_ocr_error") or default_error
        return changes

    return ingest_cancellation._change(brain_id, root_id, transition)


def maybe_finish_remote_ocr(brain_id, root_id, *, rejected=False):
    run = expire_remote_ocr(brain_id, root_id, rejected=rejected)
    if run and run.get("status") == "staging" and run.get("sealed"):
        closed = close_staging(brain_id, root_id)
        return closed or run
    return run


def record_ocr_page_failure(
    brain_id, root_id, error=None, page_index=None, total_pages=None,
):
    """Keep accepted pages. Fail the run only when nothing usable has arrived."""

    def transition(run):
        if run["status"] != "staging":
            return None
        pages = dict(run.get("ocr_pages") or {})
        key = _ocr_page_key(page_index, pages)
        if pages.get(key) in {"accepted", "failed"}:
            now, changes = _ocr_touch(run, total_pages)
            return _apply_ocr_close(changes, run, now)
        pages[key] = "failed"
        now, changes = _ocr_touch(run, total_pages)
        changes["ocr_pages"] = pages
        accepted, failed = _ocr_counts({**run, "ocr_pages": pages})
        changes["ocr_pages_accepted"] = accepted
        changes["ocr_pages_failed"] = failed
        if error:
            changes["last_ocr_error"] = error
        merged = {**run, **changes}
        if not accepted and not is_soft_ocr_page_failure(error):
            total = merged.get("ocr_total_pages")
            if not total or failed >= int(total):
                changes["status"] = "failed"
                changes["error"] = error or REMOTE_OCR_REJECTED_ERROR
                return changes
        return _apply_ocr_close(changes, run, now)

    run = ingest_cancellation._change(brain_id, root_id, transition)
    if run and run.get("status") == "failed":
        _mark_remote_ocr_failed(
            brain_id, root_id, run.get("error") or error or REMOTE_OCR_REJECTED_ERROR
        )
        return run
    if run:
        _sync_ocr_task_status(brain_id, root_id, run)
    maybe_finish_remote_ocr(brain_id, root_id)
    return ingest_cancellation._runs(brain_id).find_one({"_id": root_id}) or run


def _finish_cached_ocr(cached, status, error, extra=None):
    cached.update(extra or {})
    cached["status"] = status
    cached["stage"] = status
    cached["error"] = error
    cached.pop("remote_ocr_deadline", None)
    return cached


def fail_expired_remote_ocr(brain_id, task_id, cached=None, *, rejected=False):
    cached = dict(cached or {})
    try:
        run = maybe_finish_remote_ocr(brain_id, task_id, rejected=rejected)
    except Exception:
        log.debug("Could not expire remote OCR run %s", task_id, exc_info=True)
        run = None
    status = cached.get("status")
    if status in TERMINAL_TASK_STATUSES:
        return cached
    from src.core.saving.unstaged_ocr import settle_expired

    settled = settle_expired(brain_id, task_id, cached)
    if settled is not None:
        return settled
    if _should_report_ocr_partial(run, cached):
        source = dict(run or cached)
        source["ocr_partial"] = True
        updated = _sync_ocr_task_status(brain_id, task_id, source)
        extra, accepted, failed = _ocr_status_extra(source)
        return updated or _finish_cached_ocr(
            cached,
            "partial_failed",
            _ocr_partial_error(accepted, failed, source.get("ocr_total_pages")),
            extra,
        )
    if (status and status != "queued") or has_accepted_ocr_pages(run, cached):
        return cached
    error = None
    if run and run.get("callback_deadline") and run.get("status") == "failed":
        error = run.get("error") or REMOTE_OCR_TIMEOUT_ERROR
    deadline = cached.get("remote_ocr_deadline")
    if error is None and deadline is not None and deadline <= time():
        error = REMOTE_OCR_TIMEOUT_ERROR
    if error is None and rejected:
        error = REMOTE_OCR_REJECTED_ERROR
    if error is None:
        return cached
    return _mark_remote_ocr_failed(brain_id, task_id, error) or _finish_cached_ocr(
        cached, "failed", error,
    )


def create_batch_run(brain_id, root_id, messages):
    """Persist all batch dispatch intents and seal the run in one write."""
    units = {
        uuid5(NAMESPACE_URL, f"brainapi:ingest:{root_id}:batch:{index}").hex: {
            "state": "pending",
            "ordinal": index,
            "message": message,
        }
        for index, message in enumerate(messages)
    }
    ingest_cancellation._runs(brain_id).update_one(
        {"_id": root_id},
        {"$setOnInsert": {
            "status": "staging",
            "staged": True,
            "revision": "",
            "units": units,
            "publication_cursor": 0,
            "sealed": True,
        }},
        upsert=True,
    )
