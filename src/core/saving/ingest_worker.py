"""Execute one staged ingestion unit under a renewable claim."""

from threading import Event, Thread

from src.core.saving import ingest_staging
from src.core.saving.ingest_staged_output import commit_output
from src.core.saving.ingest_write_buffer import capture_writes


def execute_staged_unit(brain_id, root_id, unit_id, process):
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
            process()
        if not ingest_staging.renew_claim(brain_id, root_id, unit_id, attempt):
            return False
        return commit_output(brain_id, root_id, unit_id, attempt, operations)
    finally:
        stop.set()
        heartbeat.join()
