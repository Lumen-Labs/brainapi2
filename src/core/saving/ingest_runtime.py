"""Reconciliation runtime for workers sharing one deployment's lock directory."""

import logging

from src.core.saving import ingest_access_gate, ingest_cancellation, ingest_staging
from src.core.saving.ingest_publication import publish_staged_run
from src.core.saving.ingest_recovery import cleanup_abandoned_staging, reconcile_run
from src.core.saving.ingest_worker import dispatch_staged_source

log = logging.getLogger(__name__)


def recover_brain(brain_id, data_client, vector_client, graph_client):
    def publish(brain, root):
        publish_staged_run(
            brain, root, ingest_access_gate.publication_gate, data_client, vector_client, graph_client
        )

    runs = ingest_cancellation._runs(brain_id)
    active = {"staging", "publishing", "discarding"}
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
            reconcile_run(brain_id, run["_id"], dispatch_staged_source, publish)
        except ingest_access_gate.BrainPublicationBusy:
            continue
        except Exception:
            log.exception("Staged ingestion reconciliation failed")

    leftover = [
        *runs.find({"pending_cleanup": True}),
        *runs.find({"status": "rollback_failed"}),
    ]
    seen = set()
    for run in leftover:
        root_id = run["_id"]
        if root_id in seen or not ingest_staging.needs_abandoned_cleanup(run):
            continue
        seen.add(root_id)
        try:
            cleanup_abandoned_staging(brain_id, root_id)
        except Exception:
            log.exception(
                "Staged ingestion leftover cleanup failed for %s in brain %s",
                root_id, brain_id,
            )
