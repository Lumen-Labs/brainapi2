"""Reconcile one brain's durable staging state independently of broker delivery."""

from src.core.saving import ingest_access_gate, ingest_cancellation, ingest_staging
from src.core.saving.ingest_staged_output import discard_outputs

def reconcile_run(brain_id, root_id, dispatch, publish):
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
        publish(brain_id, root_id)
    elif run["status"] == "discarding":
        with ingest_access_gate.publication_gate(brain_id, root_id):
            try:
                discard_outputs(brain_id, root_id)
            except Exception:
                if not ingest_staging.record_rollback_failure(brain_id, root_id):
                    raise
                return
            ingest_staging.finish_discard(brain_id, root_id)
