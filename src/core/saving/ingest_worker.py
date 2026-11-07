"""Execute one staged ingestion unit under a renewable claim."""

from threading import Event, Thread

from src.core.saving import ingest_staging
from src.core.saving.ingest_staged_output import commit_output
from src.core.saving.ingest_write_buffer import capture_writes


def execute_staged_unit(brain_id, root_id, unit_id, process, *, discovery=False):
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
        with capture_writes(brain_id) as operations:
            result = process()
        if not ingest_staging.renew_claim(brain_id, root_id, unit_id, attempt):
            return False
        if discovery:
            if operations:
                raise ValueError("Discovery must not stage mutations")
            return ingest_staging.complete_discovery(brain_id, root_id, unit_id, attempt, result)
        return commit_output(brain_id, root_id, unit_id, attempt, operations)
    finally:
        stop.set()
        heartbeat.join()


def dispatch_staged_source(brain_id, root_id, unit_id, message):
    """Send a registered source unit. Recovery can repeat the same dispatch."""
    from src.workers.tasks.ingestion import ingest_data, ingest_url

    payload = {**message["payload"], "brain_id": brain_id}
    task = ingest_url if message.get("kind") == "url_discovery" else ingest_data
    task.apply_async(
        args=[payload],
        task_id=unit_id,
        headers={**message.get("headers", {}), "staged_ingestion_root": root_id},
    )
