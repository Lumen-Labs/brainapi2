"""Reconciliation runtime for workers sharing one deployment's lock directory."""

import logging
import os
from functools import partial

from src.core.saving import ingest_access_gate, ingest_cancellation
from src.core.saving.ingest_publication import publish_staged_run
from src.core.saving.ingest_recovery import reconcile_run
from src.core.saving.ingest_worker import dispatch_staged_source

log = logging.getLogger(__name__)


def _prepare_atomic_publication(brain_id, root_id, run, gate, data, vectors, graph):
    if gate and gate.get("publication"):
        requested = gate["publication"] == root_id and bool(gate.get("atomic"))
    else:
        requested = (
            os.getenv("INGEST_POSTGRES_ATOMIC_PUBLICATION") == "1"
            and run.get("publication_cursor", 0) == 0
        )
    if not requested:
        return False

    from src.constants.embeddings import EMBEDDING_STORES_SIZES
    from src.lib.postgresql.data import PostgreSQLDataClient
    from src.lib.postgresql.networkx_client import NetworkXGraphClient
    from src.lib.postgresql.vectors import PostgreSQLVectorStoreClient

    if not (
        isinstance(data, PostgreSQLDataClient)
        and isinstance(vectors, PostgreSQLVectorStoreClient)
        and isinstance(graph, NetworkXGraphClient)
    ):
        return False
    if gate and gate.get("publication"):
        # The durable marker was written only after schema preparation completed.
        return True
    # Prepare all DDL before advertising that readers can use committed snapshots.
    data._ensure_brain_schema(brain_id)
    graph._store._ensure_brain_schema(brain_id)
    for store in EMBEDDING_STORES_SIZES:
        vectors._ensure_store(store, brain_id)
    from src.services.explore.store import atlas_store

    atlas_store().prepare(brain_id)
    return True


def recover_brain(brain_id, data_client, vector_client, graph_client):
    def publish(brain, root):
        run = runs.find_one({"_id": root})
        if not run or run["status"] != "publishing":
            return
        gate = ingest_access_gate._gates(brain).find_one({"_id": "publication"})
        atomic = _prepare_atomic_publication(
            brain, root, run, gate, data_client, vector_client, graph_client
        )
        publish_staged_run(
            brain, root, partial(ingest_access_gate.publication_gate, atomic=atomic),
            data_client, vector_client, graph_client,
        )

    runs = ingest_cancellation._runs(brain_id)
    active = {"staging", "publishing", "discarding"}
    gate = ingest_access_gate._gates(brain_id).find_one({"_id": "publication"})
    owner = gate.get("publication") if gate else None
    if owner:
        run = runs.find_one({"_id": owner})
        if run and run["status"] in {"completed", "cancelled", "failed", "rollback_failed"}:
            try:
                with ingest_access_gate.publication_gate(brain_id, owner, atomic=bool(gate.get("atomic"))):
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
