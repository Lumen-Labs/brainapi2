"""On-demand and incremental conflict-detection jobs."""

from src.workers.app import ingestion_app


@ingestion_app.task(
    bind=True,
    name="src.workers.tasks.conflicts.scan_conflicts",
    acks_late=True,
    reject_on_worker_lost=True,
)
def scan_conflicts(self, args: dict | None = None):
    payload = args or {}
    brain_id = payload.get("brain_id") or "default"
    use_llm = payload.get("use_llm")
    from src.core.saving.conflicts import detect_conflicts
    from src.services.data.main import data_adapter
    from src.services.kg_agent.main import graph_adapter
    from src.workers.tasks.ingestion import set_ingestion_task_status

    set_ingestion_task_status(
        self.request.id, brain_id, "started", stage="conflict_scan"
    )
    stored = detect_conflicts(
        data_adapter,
        graph_adapter,
        brain_id,
        use_llm=use_llm,
    )
    set_ingestion_task_status(
        self.request.id,
        brain_id,
        "completed",
        stage="conflict_scan",
        counts={"conflicts": len(stored)},
    )
    return {
        "brain_id": brain_id,
        "count": len(stored),
        "conflict_ids": [item.id for item in stored],
    }


@ingestion_app.task(
    bind=True,
    name="src.workers.tasks.conflicts.detect_conflicts_incremental",
    acks_late=True,
    reject_on_worker_lost=True,
)
def detect_conflicts_incremental(self, args: dict | None = None):
    payload = args or {}
    brain_id = payload.get("brain_id") or "default"
    fact_ids = list(payload.get("fact_ids") or [])
    from src.core.saving.conflicts import detect_conflicts
    from src.services.data.main import data_adapter
    from src.services.kg_agent.main import graph_adapter

    stored = detect_conflicts(
        data_adapter,
        graph_adapter,
        brain_id,
        fact_ids=fact_ids,
    )
    return {
        "brain_id": brain_id,
        "count": len(stored),
        "conflict_ids": [item.id for item in stored],
    }
