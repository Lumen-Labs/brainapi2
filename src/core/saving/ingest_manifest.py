"""Durable URL manifests. Customer projections never contain source text or costs."""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import time
from datetime import datetime, timezone
from uuid import uuid4

VERSION = "sections-v1"
TERMINAL = {"completed", "failed", "partial_failed"}


def collection(brain_id):
    from src.services.data.main import data_adapter
    from src.core.saving.pipeline_store import collection as pipeline_collection
    return pipeline_collection(brain_id, "ingest_manifests")


def sections_collection(brain_id):
    from src.services.data.main import data_adapter
    from src.core.saving.pipeline_store import collection as pipeline_collection
    col = pipeline_collection(brain_id, "ingest_sections")
    col.create_index([("import_id", 1), ("ordinal", 1)])
    col.create_index("task_id", unique=True)
    return col


def verify_control(brain_id, import_id, action, token):
    secret = os.getenv("USAGE_CONTEXT_SECRET", "")
    try:
        stamp, signature = token.split(".", 1)
        if not secret or abs(time.time() - int(stamp)) > 60:
            return False
        payload = f"ingest-control:{brain_id}:{import_id}:{action}:{stamp}"
        expected = hmac.new(secret.encode(), payload.encode(), hashlib.sha256).hexdigest()
        return hmac.compare_digest(signature, expected)
    except (ValueError, AttributeError):
        return False


def create_manifest(brain_id, import_id, pages, args):
    manifests = collection(brain_id)
    if manifests.find_one({"_id": import_id}):
        return
    section_col = sections_collection(brain_id)
    page_rows = []
    seen = set()
    ordinal = 0
    for page in pages:
        url = page.canonical_url or page.url
        if url in seen:
            continue
        seen.add(url)
        page_id = hashlib.sha256(url.encode()).hexdigest()[:24]
        parts = page.sections
        if not parts and page.text.strip() and not page.skip_reason:
            from src.adapters.interfaces.crawler import CrawledSection
            parts = [CrawledSection(title="Main content", text=page.text.strip())]
        page_rows.append({"id": page_id, "url": url, "title": page.title,
            "status": "queued" if parts else "skipped",
            "reason": page.skip_reason or (None if parts else "No readable content")})
        for index, part in enumerate(parts):
            section_id = hashlib.sha256(f"{import_id}:{page_id}:{index}".encode()).hexdigest()[:32]
            section_col.update_one({"_id": section_id}, {"$setOnInsert": {
                "import_id": import_id, "page_id": page_id, "ordinal": ordinal,
                "title": part.title, "heading_path": part.heading_path,
                "text": part.text, "chars": len(part.text), "url": url,
                "page_title": page.title, "status": "pending", "stage": "queued",
                "task_id": str(uuid4()), "attempt": 1,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }}, upsert=True)
            ordinal += 1
    manifests.update_one({"_id": import_id}, {"$setOnInsert": {
        "version": VERSION, "pages": page_rows, "total": ordinal,
        "staged": bool(args.get("staged")),
        "skip_enrichment": bool(args.get("skip_enrichment")),
        "configuration": args.get("pipeline_configuration"),
        "group_tracking": args.get("group_tracking") or import_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }}, upsert=True)


def manifest_snapshot(brain_id, import_id, offset=0, limit=50, internal=False):
    manifest = collection(brain_id).find_one({"_id": import_id})
    if not manifest:
        return None
    rows = list(sections_collection(brain_id).find({"import_id": import_id}).sort("ordinal", 1))
    now = datetime.now(timezone.utc)
    for row in rows:
        if not internal or row["status"] in TERMINAL or row["status"] == "pending" or not row.get("updated_at"):
            continue
        try:
            stale = (now - datetime.fromisoformat(row["updated_at"])).total_seconds() > 86400
        except (ValueError, TypeError):
            stale = False
        if stale:
            result = sections_collection(brain_id).update_one({"_id":row["_id"],"status":row["status"],"updated_at":row["updated_at"]},{"$set":{"status":"failed","stage":"failed","error":"Worker stopped reporting for 24 hours"}})
            if result.modified_count == 1:
                row.update(status="failed",stage="failed",error="Worker stopped reporting for 24 hours")
    staged_state = None
    cancellation = {}
    if manifest.get("staged"):
        from src.core.saving import ingest_cancellation

        run = ingest_cancellation._runs(brain_id).find_one({"_id": import_id})
        if run:
            staged_state = ingest_cancellation.cancellation_status(run)
            if run.get("mode"):
                cancellation["cancellation_mode"] = run["mode"]
            elif staged_state == "staging":
                cancellation["cancellation_modes"] = ["keep_saved", "rollback"]
            units = run.get("units", {})
            for row in rows:
                unit = units.get(row["task_id"])
                if unit and unit["state"] == "complete":
                    row.update(status="completed", stage="completed")
                elif staged_state in {"cancelled", "failed", "rollback_failed"}:
                    row.update(status=staged_state, stage=staged_state)
                elif row["status"] in TERMINAL:
                    row.update(status="started", stage="persisting")
    done = sum(row["status"] == "completed" for row in rows)
    failed = sum(row["status"] in {"failed", "partial_failed"} for row in rows)
    active = any(row["status"] not in TERMINAL for row in rows)
    status = "started" if active else "partial_failed" if failed and done else "failed" if failed or not rows else "completed"
    if staged_state is not None:
        status = "started" if staged_state in {"staging", "publishing", "discarding"} else staged_state
    fields = ("page_id", "title", "heading_path", "chars", "status", "stage", "task_id", "attempt", "updated_at")
    page_rows = []
    for page in manifest["pages"]:
        children = [row for row in rows if row.get("page_id") == page["id"]]
        succeeded = sum(row["status"] == "completed" for row in children)
        errors = sum(row["status"] in {"failed", "partial_failed"} for row in children)
        running = next((row for row in children if row["status"] not in TERMINAL and row["status"] != "pending"), None)
        page_status = "started" if running else "pending" if any(row["status"] == "pending" for row in children) else "partial_failed" if errors and succeeded else "failed" if errors else "completed" if children else "skipped"
        page_rows.append({**page, "status":page_status,"stage":running.get("stage") if running else None,"progress":{"succeeded":succeeded,"failed":errors,"total":len(children)}})
    public_rows = []
    for row in rows[offset:offset + limit]:
        item = {"id": row["_id"], **{key: row.get(key) for key in fields}}
        if row.get("error"):
            item["error"] = "This section could not be processed"
        if internal:
            item["cost"] = row.get("cost")
        public_rows.append(item)
    return {"version": VERSION, "task_id": import_id, "status": status, "stage": "ingesting", **cancellation,
        "pages": page_rows, "sections": public_rows,
        "progress": {"done": done + failed, "total": len(rows), "succeeded": done, "failed": failed},
        "characters_completed": sum(row["chars"] for row in rows if row["status"] == "completed"),
        "next_offset": offset + limit if offset + limit < len(rows) else None}


def update_section(brain_id, task_id, status, stage=None, cost=None, error=None):
    col = sections_collection(brain_id)
    row = col.find_one({"task_id": task_id})
    if not row:
        return
    patch = {"status": status, "updated_at": datetime.now(timezone.utc).isoformat()}
    if stage:
        patch["stage"] = stage
    if cost:
        patch["cost"] = cost
    if error:
        patch["error"] = str(error)[:500]
    col.update_one({"_id": row["_id"], "status":{"$nin":list(TERMINAL)}}, {"$set": patch})


def _dispatch_sections_locked(brain_id, import_id, section_ids, routing=None):
    from src.workers.tasks.ingestion import ingest_data
    manifest = collection(brain_id).find_one({"_id": import_id})
    if not manifest:
        raise ValueError("Import not found")
    if manifest.get("staged"):
        from src.core.saving import ingest_cancellation, ingest_staging
        from src.core.saving.ingest_worker import dispatch_staged_source

        run = ingest_cancellation._runs(brain_id).find_one({"_id": import_id})
        if not run or run["status"] != "staging" or "discovered" not in run:
            return []
    col = sections_collection(brain_id)
    active = col.count_documents({"import_id": import_id, "status": {"$in": ["queued", "started", "persisting", "consolidating"]}})
    if routing:
        prior = manifest.get("configuration")
        if prior and prior.get("version") != routing.get("version"):
            raise ValueError("Import configuration is already pinned")
        collection(brain_id).update_one({"_id": import_id, "configuration": {"$exists": False}}, {"$set": {"configuration": routing}})
    configuration = manifest.get("configuration") or routing or {}
    # Validate all requested routes before scheduling any work.
    from src.core.saving.pipeline_runtime import route_client
    for route in configuration.get("routes", []):
        if route.get("stage", "").startswith("ingestion."):
            route_client(route["provider"], route["model"], route.get("region"))
    slots = max(0, 2 - active)
    dispatched = []
    for section_id in section_ids[:slots]:
        row = col.find_one_and_update({"_id": section_id, "import_id": import_id, "status": "pending"},
            {"$set": {"status": "queued", "updated_at": datetime.now(timezone.utc).isoformat()}})
        if not row:
            continue
        payload = {"brain_id": brain_id, "data": {"data_type": "text", "text_data": row["text"]},
            "meta_keys": {"source_url": row["url"], "source_title": row.get("page_title") or "",
                "section_title": row["title"], "heading_path": " / ".join(row.get("heading_path") or []),
                "import_id": import_id, "section_id": row["_id"]},
            "group_tracking": manifest["group_tracking"], "skip_enrichment": manifest["skip_enrichment"]}
        try:
            context = {"brain_id":brain_id, "import_id":import_id, "section_id":row["_id"], "task_id":row["task_id"], "attempt":row.get("attempt", 1), "configuration":configuration}
            if manifest.get("staged"):
                unit = ingest_staging.register_unit(
                    brain_id, import_id, row["_id"],
                    {"payload": payload, "headers": {"pipeline_operation": context}},
                    task_id=row["task_id"],
                )
                if unit is None:
                    col.update_one({"_id": row["_id"], "status": "queued"}, {"$set": {"status": "pending"}})
                    continue
                dispatch_staged_source(brain_id, import_id, unit)
            else:
                ingest_data.apply_async(args=[payload], task_id=row["task_id"], headers={"pipeline_operation":context})
            dispatched.append(section_id)
        except Exception:
            # A failed publish can be retried with the same task identity.
            col.update_one({"_id": row["_id"], "status": "queued"}, {"$set": {"status": "pending"}})
            raise
    if manifest.get("staged") and not col.count_documents({"import_id": import_id, "status": "pending"}):
        ingest_staging.seal_units(brain_id, import_id)
    return dispatched


def retry_sections(brain_id, import_id, section_ids):
    manifest = collection(brain_id).find_one({"_id": import_id})
    if manifest and manifest.get("staged"):
        # Staged retries are owned by the durable unit claim and use the same task ID.
        return
    col = sections_collection(brain_id)
    for row in col.find({"import_id": import_id, "_id": {"$in": section_ids}, "status": {"$in": ["failed", "partial_failed"]}}):
        col.update_one({"_id": row["_id"], "task_id": row["task_id"]}, {"$set": {
            "status": "pending", "stage": "queued", "task_id": str(uuid4()),
            "attempt": row.get("attempt", 1) + 1}, "$unset": {"error": "", "cost": "", "execution_claimed":""}})


def verify_usage_context(token):
    import base64
    try:
        version, encoded, signature = token.split(".")
        secret = os.getenv("USAGE_CONTEXT_SECRET", "")
        if version != "v1" or not secret:
            return False
        expected = base64.urlsafe_b64encode(hmac.new(secret.encode(), f"v1.{encoded}".encode(), hashlib.sha256).digest()).decode().rstrip("=")
        body = json.loads(base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4)))
        return hmac.compare_digest(signature, expected) and body.get("exp", 0) >= time.time() and body.get("iat", 0) <= time.time() + 5 and bool(body.get("jobId"))
    except (ValueError, TypeError):
        return False


_runtime_tokens = {}

def install_task_context():
    from celery.signals import task_prerun, task_postrun
    from src.core.saving.pipeline_runtime import bind_operation, reset_operation
    @task_prerun.connect(weak=False)
    def begin(task_id=None, task=None, **kwargs):
        value = (getattr(task.request, "headers", None) or {}).get("pipeline_operation")
        if value:
            from src.core.saving.pipeline_runtime import _stage
            _runtime_tokens[task_id] = (bind_operation(value), _stage.set("source"))
    @task_postrun.connect(weak=False)
    def end(task_id=None, **kwargs):
        token = _runtime_tokens.pop(task_id, None)
        if token is not None:
            from src.core.saving.pipeline_runtime import _stage
            reset_operation(token[0])
            _stage.reset(token[1])


def dispatch_sections(brain_id, import_id, section_ids, routing=None):
    owner = str(uuid4())
    roots = collection(brain_id)
    lock = roots.find_one_and_update({"_id":import_id, "$or":[{"dispatch_until":{"$lt":time.time()}},{"dispatch_until":{"$exists":False}}]},
        {"$set":{"dispatch_owner":owner,"dispatch_until":time.time()+120}})
    if not lock:
        return []
    try:
        return _dispatch_sections_locked(brain_id, import_id, section_ids, routing)
    finally:
        roots.update_one({"_id":import_id,"dispatch_owner":owner},{"$unset":{"dispatch_owner":"","dispatch_until":""}})


def claim_section_execution(brain_id, task_id, redelivered=False):
    col = sections_collection(brain_id)
    row = col.find_one({"task_id":task_id})
    if not row:
        return True
    if row["status"] in TERMINAL:
        return False
    query = {"task_id":task_id, "status":{"$nin":list(TERMINAL)}}
    if not redelivered:
        query["execution_claimed"] = {"$ne":True}
    claimed = col.update_one(query,{"$set":{"execution_claimed":True}})
    return claimed.modified_count == 1 or (redelivered and claimed.matched_count == 1)


def pipeline_context(request, brain_id, task_id):
    """Verify the platform envelope independently of proxy attribution headers."""
    import base64
    from fastapi import HTTPException
    raw = request.headers.get("X-Pipeline-Context", "")
    if not raw:
        return None
    try:
        version, encoded, signature = raw.split(".")
        secret = os.getenv("USAGE_CONTEXT_SECRET", "")
        expected = hmac.new(secret.encode(), f"pipeline-v1.{encoded}".encode(), hashlib.sha256).hexdigest()
        if not secret or version != "pipeline-v1" or not hmac.compare_digest(signature, expected):
            raise ValueError()
        data = json.loads(base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4)))
        if data["brain_id"] != brain_id or data["exp"] < time.time() or data["exp"] > time.time()+65:
            raise ValueError()
        return {"brain_id":brain_id,"import_id":task_id,"section_id":task_id,"task_id":task_id,"platform_job_id":data["job_id"],"configuration":data["configuration"]}
    except (ValueError, KeyError, TypeError):
        raise HTTPException(403, "Invalid pipeline context")
