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
    payload_out = stored.to_dict() if hasattr(stored, "to_dict") else {
        "count": len(stored),
        "conflict_ids": [item.id for item in stored],
    }
    set_ingestion_task_status(
        self.request.id,
        brain_id,
        "completed",
        stage="conflict_scan",
        counts={
            "conflicts": payload_out.get("count", len(stored)),
            "truncated": payload_out.get("truncated", False),
            "scanned_relationships": payload_out.get("scanned_relationships", 0),
            "total_relationships": payload_out.get("total_relationships"),
            "llm_truncated": payload_out.get("llm_truncated", False),
            "stale_resolved": payload_out.get("stale_resolved", 0),
        },
    )
    return {"brain_id": brain_id, **payload_out}


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
    subject_uuids = list(payload.get("subject_uuids") or [])
    from src.core.saving.conflicts import detect_conflicts
    from src.services.data.main import data_adapter
    from src.services.kg_agent.main import graph_adapter

    stored = detect_conflicts(
        data_adapter,
        graph_adapter,
        brain_id,
        fact_ids=fact_ids or None,
        subject_uuids=subject_uuids or None,
    )
    payload_out = stored.to_dict() if hasattr(stored, "to_dict") else {
        "count": len(stored),
        "conflict_ids": [item.id for item in stored],
    }
    return {"brain_id": brain_id, **payload_out}
