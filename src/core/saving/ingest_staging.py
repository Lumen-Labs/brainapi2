"""State transitions for imports whose workers write only to staging."""

from time import time
from uuid import NAMESPACE_URL, uuid4, uuid5

from src.core.saving import ingest_cancellation

CLAIM_SECONDS = 120
DISPATCH_RETRY_SECONDS = 30
MAX_UNIT_FAILURES = 3
MAX_ROLLBACK_FAILURES = 5


def create_run(brain_id, root_id, *, callback_deadline=None, upload_deadline=None, message=None, discovery=False):
    """Create the run and, for direct submissions, its first dispatch intent."""
    unit_id = None
    units = {}
    if message is not None:
        key = "discovery" if discovery else "source"
        unit_id = uuid5(NAMESPACE_URL, f"brainapi:ingest:{root_id}:{key}").hex
        units[unit_id] = {"state": "pending", "ordinal": 0, "message": message}
    ingest_cancellation._runs(brain_id).update_one(
        {"_id": root_id},
        {"$setOnInsert": {
            "status": "staging",
            "staged": True,
            "revision": "",
            "units": units,
            "publication_cursor": 0,
            "sealed": message is not None and not discovery,
            "callback_deadline": callback_deadline,
            "upload_deadline": upload_deadline,
        }},
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
    return bool(unit and unit.get("attempt") == token and unit.get("output_ref") == output_ref)


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
    return sorted(
        (unit for unit in run["units"].values() if unit["state"] == "complete"),
        key=lambda unit: unit["ordinal"],
    )


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
        "operation_id": uuid5(NAMESPACE_URL, f"brainapi:publish:{root_id}:{cursor}").hex,
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
        return {
            "status": "cancelled" if run.get("mode") else "completed",
            "units": _terminal_units(run),
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


def record_rollback_failure(brain_id, root_id):
    """Report exhausted cleanup retries without holding the brain gate indefinitely."""
    def transition(run):
        if run["status"] != "discarding" or run.get("mode") != "rollback":
            return None
        failures = run.get("rollback_failures", 0) + 1
        changes = {"rollback_failures": failures}
        if failures >= MAX_ROLLBACK_FAILURES:
            changes.update(status="rollback_failed", units=_terminal_units(run))
        return changes

    run = ingest_cancellation._change(brain_id, root_id, transition)
    return bool(run and run["status"] == "rollback_failed")


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


def expire_remote_ocr(brain_id, root_id, *, rejected=False):
    def transition(run):
        deadline = run.get("callback_deadline")
        if (
            run["status"] == "staging" and deadline and not run["units"]
            and (rejected or deadline <= time())
        ):
            return {"status": "failed"}
        return None

    return ingest_cancellation._change(brain_id, root_id, transition)


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
