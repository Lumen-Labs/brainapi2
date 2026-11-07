"""State transitions for imports whose workers write only to staging."""

from time import time
from uuid import NAMESPACE_URL, uuid4, uuid5

from src.core.saving import ingest_cancellation

CLAIM_SECONDS = 120


def create_run(brain_id, root_id):
    ingest_cancellation._runs(brain_id).update_one(
        {"_id": root_id},
        {"$setOnInsert": {
            "status": "staging",
            "revision": "",
            "units": {},
            "publication_cursor": 0,
        }},
        upsert=True,
    )


def register_unit(brain_id, root_id, key, message):
    """Persist a dispatch intent before sending its broker message."""
    unit_id = uuid5(NAMESPACE_URL, f"brainapi:ingest:{root_id}:{key}").hex

    def transition(run):
        if run["status"] != "staging" or unit_id in run["units"]:
            return None
        return {"units": {
            **run["units"],
            unit_id: {"state": "pending", "message": message},
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
        return {"state": "complete", "attempt": token, "output_ref": output_ref}

    unit = _update_unit(brain_id, root_id, unit_id, complete)
    return bool(unit and unit.get("attempt") == token and unit.get("output_ref") == output_ref)


def pending_dispatches(brain_id, root_id):
    run = ingest_cancellation._runs(brain_id).find_one({"_id": root_id})
    if not run or run["status"] != "staging":
        return []
    now = time()
    return [
        (unit_id, unit["message"])
        for unit_id, unit in run["units"].items()
        if unit["state"] != "complete" and unit.get("expires_at", 0) <= now
    ]


def close_staging(brain_id, root_id, mode=None):
    """Freeze the set of completed units before publishing or discarding."""
    if mode not in (None, "keep_saved", "rollback"):
        raise ValueError("Unknown cancellation mode")

    def transition(run):
        if run["status"] != "staging":
            return None
        if mode is None and (
            not run["units"] or any(unit["state"] != "complete" for unit in run["units"].values())
        ):
            return None
        return {"status": "discarding" if mode == "rollback" else "publishing", "mode": mode}

    return ingest_cancellation._change(brain_id, root_id, transition)
