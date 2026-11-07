"""Execute one staged ingestion unit under a renewable claim."""

from threading import Event, Thread

from celery.exceptions import Ignore

from src.core.saving import ingest_staging
from src.core.saving.ingest_staged_output import commit_output
from src.core.saving.ingest_write_buffer import capture_writes


def execute_staged_unit(brain_id, root_id, unit_id, process, *, discovery=False, wait_for_sections=False):
    attempt = ingest_staging.claim_unit(brain_id, root_id, unit_id)
    if attempt is None:
        return False
    stop = Event()

    def renew():
        while not stop.wait(ingest_staging.CLAIM_SECONDS / 3):
            if not ingest_staging.renew_claim(brain_id, root_id, unit_id, attempt):
                return

    heartbeat = Thread(target=renew, daemon=True)
    heartbeat.start()
    try:
        try:
            with capture_writes(
                brain_id,
                lambda: ingest_staging.renew_claim(brain_id, root_id, unit_id, attempt),
            ) as operations:
                result = process()
            if discovery and (operations or (not result and not wait_for_sections)):
                raise ValueError("Discovery must produce source units without staging mutations")
        except Ignore:
            return False
        except Exception:
            ingest_staging.record_unit_failure(brain_id, root_id, unit_id, attempt)
            raise
        if not ingest_staging.renew_claim(brain_id, root_id, unit_id, attempt):
            return False
        if discovery:
            return ingest_staging.complete_discovery(
                brain_id, root_id, unit_id, attempt, result, wait_for_sections=wait_for_sections,
            )
        return commit_output(brain_id, root_id, unit_id, attempt, operations)
    finally:
        stop.set()
        heartbeat.join()


def dispatch_staged_source(brain_id, root_id, unit_id, _message=None):
    """Send a registered source unit. Recovery can repeat the same dispatch."""
    from src.workers.tasks.ingestion import ingest_data, ingest_file, ingest_structured_data, ingest_url

    message = ingest_staging.reserve_dispatch(brain_id, root_id, unit_id)
    if message is None:
        return
    payload = {**message["payload"], "brain_id": brain_id}
    headers = {**message.get("headers", {}), "staged_ingestion_root": root_id}
    if message.get("kind") == "file_discovery":
        ingest_file.apply_async(kwargs=payload, task_id=unit_id, headers=headers)
        return
    task = {
        "source": ingest_data,
        "structured": ingest_structured_data,
        "url_discovery": ingest_url,
    }[message.get("kind", "source")]
    task.apply_async(
        args=[payload],
        task_id=unit_id,
        headers=headers,
    )
