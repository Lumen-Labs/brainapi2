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
from src.services.observations.main import observations_agent
from src.workers.app import ingestion_app
from src.constants.tasks.ingestion import (
    IngestionTaskArgs,
    IngestionTaskDataType,
)


@ingestion_app.task(bind=True)
def ingest_data(self, args: dict):
    """
    Ingest data into the database.
    """
    payload = IngestionTaskArgs(**args)

    # ================================================
    # --------------- Data Saving --------------------
    # ================================================
    pass

    # ================================================
    # --------------- Observations -------------------
    # ================================================
    observations = observations_agent.observe(
        text=(
            payload.data.text_data
            if payload.data.data_type == IngestionTaskDataType.TEXT.value
            else json.dumps(payload.data.json_data)
        ),
        observate_for=payload.observate_for,
    )

    # ================================================
    # ------------ Triplet Extraction ----------------
    # ================================================
    information = f"""
    {payload.data.text_data
            if payload.data.data_type == IngestionTaskDataType.TEXT.value
            else json.dumps(payload.data.json_data)}
            
    Annotations:
    {observations}
    """

    kg_agent.update_kg(
        information=information,
        metadata=payload.meta_keys,
        identification_params=payload.identification_params,
    )

    return self.request.id
