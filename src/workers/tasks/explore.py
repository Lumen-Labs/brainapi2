"""Atlas jobs run on the existing consolidation queue, outside request workers."""

from src.workers.app import ingestion_app


@ingestion_app.task(
    name="src.workers.tasks.explore.build_atlas", soft_time_limit=780, time_limit=840
)
def build_atlas(brain_id, token):
    from src.services.explore.store import atlas_store

    atlas_store().build(brain_id, token)
