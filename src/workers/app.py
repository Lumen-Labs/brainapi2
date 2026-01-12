"""
File: /celery.py
Created Date: Sunday October 19th 2025
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Sunday October 19th 2025 9:16:36 am
Modified By: the developer formerly known as Christian Nonis at <alch.infoemail@gmail.com>
-----
"""

import os
from celery import Celery

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

ingestion_app.conf.update(
    worker_max_tasks_per_child=50,
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    worker_pool="threads",
    worker_concurrency=config.celery.worker_concurrency,
    broker_pool_limit=100,
    broker_connection_retry_on_startup=True,
    broker_connection_timeout=10,
    result_expires=3600 * 24 * 7,
    task_track_started=True,
    task_time_limit=3600,
    task_soft_time_limit=3000,
    broker_transport_options={
        "visibility_timeout": 7200,
        "fanout_prefix": True,
        "fanout_patterns": True,
    },
)
