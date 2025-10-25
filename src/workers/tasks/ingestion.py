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
from src.constants.data import Observation, TextChunk
from src.services.data.main import data_adapter
from src.services.kg_agent.main import (
    embeddings_adapter,
    kg_agent,
    vector_store_adapter,
)
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
    text_chunk = data_adapter.save_text_chunk(
        TextChunk(
            text=(
                payload.data.text_data
                if payload.data.data_type == IngestionTaskDataType.TEXT.value
                else json.dumps(payload.data.json_data)
            ),
            metadata=payload.meta_keys,
        )
    )
    text_chunk_vector = embeddings_adapter.embed_text(text_chunk.text)
    text_chunk_vector.metadata = {
        **payload.meta_keys,
        "resource_id": text_chunk.id,
    }
    vector_store_adapter.add_vectors(
        [text_chunk_vector],
        "data",
    )

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
    data_adapter.save_observations(
        [
            Observation(
                text=observation,
                metadata=payload.meta_keys,
                resource_id=text_chunk.id,
            )
            for observation in observations
        ]
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
        metadata={**payload.meta_keys, "resource_id": text_chunk.id},
        identification_params=payload.identification_params,
        preferred_entities=payload.preferred_extraction_entities,
    )

    return self.request.id
