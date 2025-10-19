"""
File: /injestion.py
Created Date: Sunday October 19th 2025
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Sunday October 19th 2025 12:14:21 pm
Modified By: the developer formerly known as Christian Nonis at <alch.infoemail@gmail.com>
-----
"""

from celery import shared_task


@shared_task
def inject_data(data: dict):
    """
    Ingest data into the database.
    """
    print(data)
