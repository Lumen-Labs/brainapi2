"""
File: /ingestion.py
Created Date: Sunday October 19th 2025
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Sunday October 19th 2025 12:14:21 pm
Modified By: the developer formerly known as Christian Nonis at <alch.infoemail@gmail.com>
-----
"""

from celery import shared_task

from src.constants.tasks.ingestion import IngestionTaskArgs


@shared_task(bind=True)
def ingest_data(self, args: IngestionTaskArgs):
    """
    Ingest data into the database.
    """
