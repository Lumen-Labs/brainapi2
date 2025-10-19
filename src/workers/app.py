"""
File: /celery.py
Created Date: Sunday October 19th 2025
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Sunday October 19th 2025 9:16:36 am
Modified By: the developer formerly known as Christian Nonis at <alch.infoemail@gmail.com>
-----
"""

from celery import Celery


injection_app = Celery(
    "injection_app",
    broker="amqp://kalo:kalo@localhost:5672//",
    backend="rpc://",
    include=["src.workers.tasks.injestion"],
)
