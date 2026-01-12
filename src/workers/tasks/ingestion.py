"""
File: /ingestion.py
Created Date: Sunday October 19th 2025
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Saturday December 13th 2025
Modified By: the developer formerly known as Christian Nonis at <alch.infoemail@gmail.com>
-----
"""

import json
import tomllib
from pathlib import Path
from uuid import uuid4
from src.constants.data import (
    KGChangeLogPredicateUpdatedProperty,
    KGChanges,
    KGChangesType,
    Observation,
    PartialNode,
    StructuredData,
    TextChunk,
    KGChangeLogNodePropertiesUpdated,
)
from src.constants.kg import Node
from src.constants.prompts.misc import NODE_DESCRIPTION_PROMPT
from src.core.saving.auto_kg import enrich_kg_from_input
from src.services.api.constants.requests import IngestionStructuredRequestBody
from src.services.data.main import data_adapter
from src.services.input.agents import llm_small_adapter
from src.services.kg_agent.main import (
    embeddings_adapter,
    graph_adapter,
    vector_store_adapter,
)
from src.services.observations.main import observations_agent
from src.workers.app import ingestion_app
from src.constants.tasks.ingestion import (
    IngestionTaskArgs,
    IngestionTaskDataType,
)
from src.services.kg_agent.main import cache_adapter

PYPROJECT_PATH = Path(__file__).parent.parent.parent.parent / "pyproject.toml"
with open(PYPROJECT_PATH, "rb") as f:
    BRAIN_VERSION = tomllib.load(f)["project"]["version"]


def format_textual_data(data: dict, include_keys: bool = True) -> str:
    def format_value(v):
        if isinstance(v, str):
            return v
        if isinstance(v, list):
            return ", ".join(str(item) for item in v)
        return str(v)

    if include_keys:
        return "\n".join(f"{k}: {format_value(v)}" for k, v in data.items())
    return "\n".join(format_value(v) for v in data.values())


@ingestion_app.task(bind=True)
def ingest_data(self, args: dict):
    """
    Ingest data into the database.
    """
    payload = None
    try:
        payload = IngestionTaskArgs(**args)

        cache_adapter.set(
            key=f"task:{self.request.id}",
            value=json.dumps({"status": "started", "task_id": self.request.id}),
            brain_id=payload.brain_id,
            expires_in=3600 * 24 * 7,
        )

        payload.meta_keys = (
            {
                f"{k.replace(' ', '_').lower()}": v
                for k, v in payload.meta_keys.items()
                if v is not None
            }
            if payload.meta_keys
            else None
        )

        payload.identification_params = (
            {
                f"{k.replace(' ', '_').lower()}": v
                for k, v in payload.identification_params.items()
                if v is not None
            }
            if payload.identification_params
            else None
        )

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
                brain_version=BRAIN_VERSION,
            ),
            brain_id=payload.brain_id,
        )
        text_chunk_vector = embeddings_adapter.embed_text(text_chunk.text)
        text_chunk_vector.metadata = {
            **(payload.meta_keys if payload.meta_keys else {}),
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
        enrich_kg_from_input(payload.data.text_data, brain_id=payload.brain_id)

        cache_adapter.set(
            key=f"task:{self.request.id}",
            value=json.dumps({"status": "completed", "task_id": self.request.id}),
            brain_id=payload.brain_id,
            expires_in=3600 * 24 * 7,
        )

        return self.request.id

    except Exception as e:
        brain_id = payload.brain_id if payload else args.get("brain_id", "default")
        error_result = {
            "status": "failed",
            "task_id": self.request.id,
            "error": str(e),
            "payload": payload.model_dump(mode="json") if payload else args,
        }

        cache_adapter.set(
            key=f"task:{self.request.id}",
            value=json.dumps(error_result),
            brain_id=brain_id,
            expires_in=3600 * 24 * 7,
        )

        raise


@ingestion_app.task(bind=True)
def ingest_structured_data(self, args: dict):
    """
    Ingest structured data into the database.
    """
    payload = None
    try:
        payload = IngestionStructuredRequestBody(**args)

        cache_adapter.set(
            key=f"task:{self.request.id}",
            value=json.dumps(
                {
                    "status": "started",
                    "task_id": self.request.id,
                    "total_elements": len(payload.data),
                }
            ),
            brain_id=payload.brain_id,
            expires_in=3600 * 24 * 7,
        )

        for element in payload.data:
            uuid = str(uuid4())
            description = llm_small_adapter.generate_text(
                prompt=NODE_DESCRIPTION_PROMPT.format(
                    element=json.dumps(element.model_dump(mode="json"), indent=2),
                    element_type=", ".join(element.types) if element.types else "thing",
                    element_name=(
                        element.identification_params.name
                        if element.identification_params
                        else "the thing"
                    ),
                )
            )
            node = graph_adapter.add_nodes(
                [
                    Node(
                        uuid=uuid,
                        labels=element.types,
                        name=element.identification_params.name,
                        description=description,
                        properties={
                            **(element.json_data if element.json_data else {}),
                            **(
                                element.identification_params.model_dump(mode="json")
                                if element.identification_params
                                else {}
                            ),
                        },
                    )
                ],
                brain_id=payload.brain_id,
                identification_params=element.identification_params.model_dump(
                    mode="json"
                ),
                metadata=element.textual_data,
            )[0]

            # Saving the node properties updated change
            kg_changes = KGChanges(
                type=KGChangesType.NODE_PROPERTIES_UPDATED,
                change=KGChangeLogNodePropertiesUpdated(
                    node=PartialNode(
                        **node.model_dump(mode="json"),
                    ),
                    properties=[
                        KGChangeLogPredicateUpdatedProperty(
                            property=property,
                            previous_value=None,
                            new_value=node.properties[property],
                        )
                        for property in node.properties
                    ],
                ),
            )
            data_adapter.save_kg_changes(kg_changes, brain_id=payload.brain_id)

            vector = embeddings_adapter.embed_text(node.name)
            vector.id = node.uuid
            vector.metadata = {
                "labels": node.labels,
                "name": node.name,
                "uuid": node.uuid,
            }
            vector_store_adapter.add_vectors(vectors=[vector], store="nodes")

            data_adapter.save_structured_data(
                StructuredData(
                    id=uuid,
                    data=element.json_data,
                    types=element.types,
                    metadata={
                        "identification_params": element.identification_params,
                        "textual_data": element.textual_data,
                        "labels": element.types,
                        "name": element.identification_params.name,
                    },
                    brain_version=BRAIN_VERSION,
                ),
                brain_id=payload.brain_id,
            )
            vector = embeddings_adapter.embed_text(
                format_textual_data(element.textual_data, include_keys=False)
                + "\n"
                + (
                    ", ".join(node.labels)
                    if isinstance(node.labels, list)
                    else node.labels
                )
                + "\n"
                + node.name,
            )
            vector.id = node.uuid
            vector.metadata = {
                "labels": node.labels,
                "name": node.name,
                "uuid": node.uuid,
            }
            vector_store_adapter.add_vectors(vectors=[vector], store="data")

            text_content = format_textual_data(element.textual_data)
            observations = observations_agent.observe(
                text=text_content,
                observate_for=payload.observate_for,
            )
            for obs in observations:
                obs_vector = embeddings_adapter.embed_text(obs)
                obs_vector.id = str(uuid4())
                obs_vector.metadata = {
                    "resource_id": node.uuid,
                    "labels": node.labels,
                    "name": node.name,
                }
                vector_store_adapter.add_vectors(
                    vectors=[obs_vector], store="observations"
                )
            if len(observations) > 0:
                data_adapter.save_observations(
                    [
                        Observation(
                            id=node.uuid,
                            text=obs,
                            metadata={
                                "labels": node.labels,
                                "name": node.name,
                            },
                            resource_id=node.uuid,
                        )
                        for obs in observations
                    ],
                    brain_id=payload.brain_id,
                )

            enrich_kg_from_input(
                element.textual_data, targeting=node, brain_id=payload.brain_id
            )

        cache_adapter.set(
            key=f"task:{self.request.id}",
            value=json.dumps({"status": "completed", "task_id": self.request.id}),
            brain_id=payload.brain_id,
            expires_in=3600 * 24 * 7,
        )

        return self.request.id

    except Exception as e:
        brain_id = payload.brain_id if payload else args.get("brain_id", "default")
        error_result = {
            "status": "failed",
            "task_id": self.request.id,
            "error": str(e),
            "payload": payload.model_dump(mode="json") if payload else args,
        }

        cache_adapter.set(
            key=f"task:{self.request.id}",
            value=json.dumps(error_result),
            brain_id=brain_id,
            expires_in=3600 * 24 * 7,
        )

        raise
