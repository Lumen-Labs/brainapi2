"""
File: /ingestion.py
Created Date: Sunday October 19th 2025
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Sunday October 19th 2025 12:14:21 pm
Modified By: the developer formerly known as Christian Nonis at <alch.infoemail@gmail.com>
-----
"""

import json
from src.services.kg_agent.main import kg_agent
from src.workers.app import ingestion_app
from src.constants.tasks.ingestion import (
    IngestionTaskArgs,
    IngestionTaskDataType,
    IngestionTaskJsonArgs,
    IngestionTaskTextArgs,
)


@ingestion_app.task(bind=True)
def ingest_data(self, args: dict):
    """
    Ingest data into the database.
    """
    payload: IngestionTaskArgs = None

    if args.get("data_type") == IngestionTaskDataType.TEXT.value:
        payload = IngestionTaskTextArgs(**args)
    elif args.get("data_type") == IngestionTaskDataType.JSON.value:
        payload = IngestionTaskJsonArgs(**args)
    else:
        raise ValueError(f"Invalid data type: {args.get('data_type')}")

    response = kg_agent.update_kg(
        (
            payload.text_data
            if payload.data_type == IngestionTaskDataType.TEXT.value
            else json.dumps(payload.json_data)
        ),
        (
            payload.meta_keys
            if payload.data_type == IngestionTaskDataType.JSON.value
            else None
        ),
    )
    print(response)
    return self.request.id
