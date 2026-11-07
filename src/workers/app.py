"""
File: /celery.py
Created Date: Sunday October 19th 2025
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Thursday February 19th 2026 7:45:12 pm
Modified By: Christian Nonis <alch.infoemail@gmail.com>
-----
"""

import os
import warnings
from pathlib import Path

import dotenv

_project_root = Path(__file__).resolve().parent.parent.parent
dotenv.load_dotenv(_project_root / ".env")

warnings.filterwarnings(
    "ignore",
    message=r"Unrecognized FinishReason enum value",
    category=UserWarning,
)

from src.workers.kombu_redis_patch import apply_kombu_redis_unblocked_patch
from src.workers.redis_url import redis_connection_url

apply_kombu_redis_unblocked_patch()

from celery import Celery
from kombu import Queue

from src.config import config
from src.core.plugins.celery_discovery import (
    default_plugins_dir,
    discover_plugin_celery,
)
from src.lib.tracing import start_runtime_monitoring
from src.lib.tracing.workers import install_celery_tracing

os.environ.setdefault("GRPC_ENABLE_FORK_SUPPORT", "1")

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = os.getenv("REDIS_PORT", "6379")
REDIS_URL = redis_connection_url(
    REDIS_HOST,
    REDIS_PORT,
    os.getenv("REDIS_PASSWORD"),
)

ingestion_app = Celery(
    "ingestion_app",
    broker=(
        "amqp://kalo:kalo@localhost:5672/"
        if os.getenv("CELERY_BACKEND") == "rabbitmq"
        else REDIS_URL
    ),
    backend=(
        "rpc://"
        if os.getenv("CELERY_BACKEND") == "rabbitmq"
        else REDIS_URL
    ),
    include=["src.workers.tasks.ingestion", "src.workers.tasks.explore"],
)

_PLUGIN_QUEUES, _PLUGIN_ROUTES = discover_plugin_celery(default_plugins_dir(_project_root))

TASK_QUEUES = (
    Queue("ingest_data", routing_key="ingest_data"),
    Queue(
        "process_architect_relationships", routing_key="process_architect_relationships"
    ),
    Queue("ingest_structured_data", routing_key="ingest_structured_data"),
    Queue("consolidate_graph", routing_key="consolidate_graph"),
    Queue("finalize_ingestion", routing_key="finalize_ingestion"),
    Queue("ingest_file", routing_key="ingest_file"),
) + _PLUGIN_QUEUES

TASK_ROUTES = {
    "src.workers.tasks.ingestion.ingest_url": {"queue": "ingest_data"},
    "src.workers.tasks.ingestion.mark_ingestion_chain_failed": {"queue": "ingest_data"},
    "src.workers.tasks.ingestion.ingest_data": {"queue": "ingest_data"},
    "src.workers.tasks.ingestion.process_architect_relationships": {
        "queue": "process_architect_relationships"
    },
    "src.workers.tasks.ingestion.ingest_structured_data": {
        "queue": "ingest_structured_data"
    },
    "src.workers.tasks.ingestion.consolidate_graph_async": {
        "queue": "consolidate_graph"
    },
    "src.workers.tasks.ingestion.finalize_ingestion_task": {
        "queue": "finalize_ingestion"
    },
    "src.workers.tasks.ingestion.ingest_file": {"queue": "ingest_file"},
    **_PLUGIN_ROUTES,
}

ingestion_app.conf.update(
    task_queues=TASK_QUEUES,
    task_routes=TASK_ROUTES,
    worker_max_tasks_per_child=50,
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    worker_pool="threads",
    worker_concurrency=int(config.celery.worker_concurrency),
    broker_pool_limit=100,
    broker_connection_retry_on_startup=True,
    broker_connection_timeout=10,
    result_expires=3600 * 24 * 7,
    task_track_started=True,
    task_time_limit=86400,
    task_soft_time_limit=86400,
    broker_transport_options={
        "visibility_timeout": 86400,
        "fanout_prefix": True,
        "fanout_patterns": True,
    },
)
install_celery_tracing(service_name="brainapi-worker")
start_runtime_monitoring(service_name="brainapi-worker")


if os.getenv("STAGED_INGESTION_ENABLED") == "1" and os.getenv("INGEST_PUBLICATION_LOCK_DIR"):
    import logging
    from threading import Event, Thread

    from celery.signals import worker_ready, worker_shutdown

    _recovery_stop = Event()

    def _recover_ingestion():
        from src.core.instances import data_adapter, graph_adapter, vector_store_adapter
        from src.core.saving.ingest_runtime import recover_brain

        while not _recovery_stop.is_set():
            try:
                for brain in data_adapter.get_brains_list():
                    recover_brain(
                        brain.name_key, data_adapter.data,
                        vector_store_adapter.vector_store, graph_adapter.graph,
                    )
            except Exception:
                logging.getLogger(__name__).exception("Could not scan staged ingestions")
            _recovery_stop.wait(5)

    @worker_ready.connect
    def _start_ingestion_recovery(**kwargs):
        _recovery_stop.clear()
        Thread(target=_recover_ingestion, name="ingestion-recovery", daemon=True).start()

    @worker_shutdown.connect
    def _stop_ingestion_recovery(**kwargs):
        _recovery_stop.set()
