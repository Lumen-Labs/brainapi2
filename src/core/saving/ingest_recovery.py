"""Reconcile one brain's durable staging state independently of broker delivery."""

import logging

from src.core.saving import ingest_access_gate, ingest_cancellation, ingest_staging
from src.core.saving.ingest_staged_output import discard_outputs

log = logging.getLogger(__name__)


def cleanup_abandoned_staging(brain_id, root_id):
    """Remove leftover private staging data after a terminal settlement failure."""
    discard_outputs(brain_id, root_id)
    ingest_staging.mark_cleanup_complete(brain_id, root_id)


def reconcile_run(brain_id, root_id, dispatch, publish, rollback=None):
    """Retry pending work. Publish must acquire the brain's publication gate."""
    run = ingest_cancellation._runs(brain_id).find_one({"_id": root_id})
    if run is None:
        return

    if run["status"] == "staging":
        if run.get("upload_deadline") and not run["units"]:
            run = ingest_staging.expire_file_upload(brain_id, root_id)
            if run is None:
                return
        if run.get("callback_deadline"):
            run = ingest_staging.expire_remote_ocr(brain_id, root_id)
            if not run or run["status"] != "staging":
                return
        for unit_id, message in ingest_staging.pending_dispatches(brain_id, root_id):
            dispatch(brain_id, root_id, unit_id, message)
        run = ingest_staging.close_staging(brain_id, root_id)
        if run is None:
            return

    if run["status"] == "publishing":
        try:
            publish(brain_id, root_id)
        except ingest_access_gate.BrainPublicationBusy:
            raise
        except Exception:
            current = ingest_cancellation._runs(brain_id).find_one({"_id": root_id})
            if current is None or current["status"] != "publishing":
                raise
            if not ingest_staging.record_settlement_failure(brain_id, root_id, "publishing"):
                raise
            log.exception(
                "Staged ingestion %s for brain %s failed permanently while publishing",
                root_id, brain_id,
            )
            run = ingest_cancellation._runs(brain_id).find_one({"_id": root_id})
            if run is None:
                return
            if run["status"] != "publish_partial":
                ingest_access_gate.release_publication(brain_id, root_id)

    if run["status"] == "publish_partial":
        with ingest_access_gate.publication_gate(brain_id, root_id):
            try:
                if rollback is not None:
                    rollback(brain_id, root_id)
            except ingest_access_gate.BrainPublicationBusy:
                raise
            except Exception:
                if not ingest_staging.record_settlement_failure(brain_id, root_id, "publish_partial"):
                    raise
                log.exception(
                    "Staged ingestion %s for brain %s failed permanently while rolling back a partial publish",
                    root_id, brain_id,
                )
                return
            ingest_staging.finish_publish_rollback(brain_id, root_id)
    elif run["status"] == "discarding":
        with ingest_access_gate.publication_gate(brain_id, root_id):
            try:
                discard_outputs(brain_id, root_id)
            except Exception:
                if not ingest_staging.record_settlement_failure(brain_id, root_id, "discarding"):
                    raise
                log.exception(
                    "Staged ingestion %s for brain %s failed permanently while discarding",
                    root_id, brain_id,
                )
                return
            ingest_staging.finish_discard(brain_id, root_id)
