"""Reconciliation runtime for workers sharing one deployment's lock directory."""

import logging

from src.core.saving import ingest_access_gate, ingest_cancellation, ingest_staging
from src.core.saving.ingest_publication import publish_staged_run, rollback_published_run
from src.core.saving.ingest_recovery import cleanup_abandoned_staging, reconcile_run
from src.core.saving.ingest_worker import dispatch_staged_source

log = logging.getLogger(__name__)


def recover_brain(brain_id, data_client, vector_client, graph_client):
    def publish(brain, root):
        publish_staged_run(
            brain, root, ingest_access_gate.publication_gate, data_client, vector_client, graph_client
        )

    def rollback(brain, root):
        rollback_published_run(brain, root, data_client, vector_client, graph_client)

    runs = ingest_cancellation._runs(brain_id)
    indexer = getattr(runs, "create_index", None)
    if callable(indexer):
        try:
            try:
                indexer("pending_cleanup", partialFilterExpression={"pending_cleanup": True})
            except TypeError:
                indexer("pending_cleanup")
        except Exception:
            log.exception("Could not create pending_cleanup index on ingest_runs for brain %s", brain_id)
    active = {"staging", "publishing", "discarding", "publish_partial"}
    gate = ingest_access_gate._gates(brain_id).find_one({"_id": "publication"})
    owner = gate.get("publication") if gate else None
    if owner:
        run = runs.find_one({"_id": owner})
        if run and run["status"] in {"completed", "cancelled", "failed", "rollback_failed"}:
            try:
                with ingest_access_gate.publication_gate(brain_id, owner):
                    pass
            except ingest_access_gate.BrainPublicationBusy:
                pass

    for run in runs.find({"status": {"$in": sorted(active)}}):
        try:
            reconcile_run(brain_id, run["_id"], dispatch_staged_source, publish, rollback)
        except ingest_access_gate.BrainPublicationBusy:
            continue
        except Exception:
            log.exception("Staged ingestion reconciliation failed")

    for run in runs.find({"pending_cleanup": True}):
        if not ingest_staging.needs_abandoned_cleanup(run):
            continue
        try:
            cleanup_abandoned_staging(brain_id, run["_id"])
        except Exception:
            log.exception(
                "Staged ingestion leftover cleanup failed for %s in brain %s",
                run["_id"], brain_id,
            )
