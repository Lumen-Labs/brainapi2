"""Reconciliation runtime for workers sharing one deployment's lock directory."""

import logging

from src.core.saving import ingest_access_gate, ingest_cancellation
from src.core.saving.ingest_publication import publish_staged_run
from src.core.saving.ingest_recovery import reconcile_run
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
