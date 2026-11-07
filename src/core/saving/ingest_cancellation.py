"""Durable cooperative cancellation for explicitly tracked ingestion roots."""

import os
from contextvars import ContextVar
from functools import wraps
from uuid import uuid4

from celery.exceptions import Ignore

from src.core.saving.pipeline_store import collection

TERMINAL = {"completed", "failed", "partial_failed", "cancelled"}
_current = ContextVar("ingestion_cancellation", default=None)


def _runs(brain_id):
    return collection(brain_id, "ingest_runs")


def enabled():
    return os.getenv("INGESTION_CANCELLATION_ENABLED") == "1"


def task_control_enabled():
    return enabled() or os.getenv("STAGED_INGESTION_ENABLED") == "1"


def register(brain_id, task_id):
    if not enabled():
        return {}
    _runs(brain_id).update_one(
        {"_id": task_id},
        {"$setOnInsert": {"status": "active", "workers": [], "revision": ""}},
        upsert=True,
    )
    return {"ingestion_root": task_id}


def _change(brain_id, task_id, transition, *, store=None):
    runs = (store or _runs)(brain_id)
    while True:
        run = runs.find_one({"_id": task_id})
        if run is None:
            return None
        changes = transition(run)
        if changes is None:
            return run
        changes["revision"] = uuid4().hex
        result = runs.update_one(
            {"_id": task_id, "revision": run["revision"]}, {"$set": changes}
        )
        if result.modified_count:
            return {**run, **changes}


class CancellationConflict(Exception):
    pass


def request_cancellation(brain_id, task_id, mode):
    def transition(run):
        if run.get("mode") == mode:
            return None
        if run["status"] == "staging":
            return {"mode": mode, "status": "discarding" if mode == "rollback" else "publishing"}
        if run["status"] != "active" or mode != "keep_saved":
            raise CancellationConflict
        return {
            "mode": mode,
            "status": "cancelling" if run["workers"] else "cancelled",
        }

    return _change(brain_id, task_id, transition)


def cancellation_status(run):
    if run.get("mode") and run["status"] in {"publishing", "discarding"}:
        return "cancelling"
    return run["status"]


def public_state(brain_id, task_id, payload):
    if not task_control_enabled():
        return payload
    run = _runs(brain_id).find_one({"_id": task_id})
    if run and run.get("mode"):
        status = cancellation_status(run)
        return {**payload, "status": status, "stage": status, "cancellation_mode": run["mode"]}
    if run and run.get("staged") and run["status"] in {"publishing", "discarding", "completed", "failed"}:
        status = "processing" if run["status"] in {"publishing", "discarding"} else run["status"]
        return {**payload, "status": status, "stage": run["status"]}
    if run and run["status"] in {"active", "staging"}:
        modes = ["keep_saved", "rollback"] if run["status"] == "staging" else ["keep_saved"]
        return {"status": "queued", **payload, "cancellation_modes": modes}
    return payload


def record_terminal(brain_id, task_id, status):
    if not enabled():
        return
    _change(
        brain_id, task_id,
        lambda run: {"status": status} if run["status"] == "active" else None,
    )


def checkpoint():
    from src.core.saving.ingest_write_buffer import check_staged_claim

    check_staged_claim()
    scope = _current.get()
    if scope and _runs(scope[0]).find_one({"_id": scope[1]})["status"] != "active":
        raise Ignore()


def chain_headers():
    scope = _current.get()
    return {"ingestion_root": scope[1]} if scope else {}


def tracked_worker(brain_from_args):
    def decorate(fn):
        @wraps(fn)
        def wrapped(task, *args, **kwargs):
            root = (getattr(task.request, "headers", None) or {}).get("ingestion_root")
            if not isinstance(root, str) or not root:
                return fn(task, *args, **kwargs)

            brain_id = brain_from_args(*args, **kwargs)
            worker = uuid4().hex
            run = _change(
                brain_id, root,
                lambda row: {"workers": [*row["workers"], worker]}
                if row["status"] == "active" else None,
            )
            if not run or worker not in run["workers"]:
                raise Ignore()

            def leave(row):
                workers = [entry for entry in row["workers"] if entry != worker]
                changes = {"workers": workers}
                if not workers and row["status"] == "cancelling":
                    changes["status"] = "cancelled"
                return changes

            token = _current.set((brain_id, root))
            try:
                return fn(task, *args, **kwargs)
            finally:
                _current.reset(token)
                _change(brain_id, root, leave)

        return wrapped

    return decorate
