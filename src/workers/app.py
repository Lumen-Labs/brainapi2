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
