"""
File: /celery.py
Created Date: Sunday October 19th 2025
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Sunday October 19th 2025 9:16:36 am
Modified By: Christian Nonis <alch.infoemail@gmail.com>
-----
"""

import os
from pathlib import Path

import dotenv

_project_root = Path(__file__).resolve().parent.parent.parent
dotenv.load_dotenv(_project_root / ".env")

from celery import Celery
from kombu import Queue

from src.config import config

os.environ.setdefault("GRPC_ENABLE_FORK_SUPPORT", "1")

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = os.getenv("REDIS_PORT", "6379")

ingestion_app = Celery(
    "ingestion_app",
    broker=(
        "amqp://kalo:kalo@localhost:5672/"
        if os.getenv("CELERY_BACKEND") == "rabbitmq"
        else f"redis://{REDIS_HOST}:{REDIS_PORT}/0"
    ),
    backend=(
        "rpc://"
        if os.getenv("CELERY_BACKEND") == "rabbitmq"
        else f"redis://{REDIS_HOST}:{REDIS_PORT}/0"
    ),
    include=["src.workers.tasks.ingestion"],
)

TASK_QUEUES = (
    Queue("ingest_data", routing_key="ingest_data"),
    Queue(
        "process_architect_relationships", routing_key="process_architect_relationships"
    ),
    Queue("ingest_structured_data", routing_key="ingest_structured_data"),
    Queue("consolidate_graph", routing_key="consolidate_graph"),
)

TASK_ROUTES = {
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
