"""
File: /ingestion.py
Created Date: Sunday October 19th 2025
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Thursday January 29th 2026 8:44:06 pm
Modified By: Christian Nonis <alch.infoemail@gmail.com>
-----
"""

import datetime
import json
import tomllib
from pathlib import Path
from uuid import uuid4
from typing import List, Optional, Tuple
from concurrent.futures import (
    Future,
    ThreadPoolExecutor,
    TimeoutError as FutureTimeoutError,
)
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
from src.constants.kg import Node, Predicate
from src.constants.agents import ArchitectAgentRelationship
from src.constants.prompts.misc import NODE_DESCRIPTION_PROMPT
from src.core.saving.auto_kg import enrich_kg_from_input
from src.core.saving.ingestion_manager import IngestionManager
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
    Ingest a payload into the system, persist its content and metadata, generate embeddings and observations, and trigger knowledge-graph enrichment.

    Parameters:
        args (dict): Raw task arguments parsed into an IngestionTaskArgs model; must include a brain_id and data payload.

    Returns:
        task_id (str): The identifier of the ingestion task (Celery request id) that was created/updated.
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
def process_architect_relationships(self, args: dict):
    """
    Process a batch of architect relationships and ingest corresponding nodes, vectors, and graph edges.

    Parameters:
        args (dict): Task payload containing:
            - "relationships" (List[dict]): List of relationship payloads convertible to ArchitectAgentRelationship.
            - "brain_id" (str, optional): Target brain identifier; defaults to "default".

    Description:
        For each relationship in `args["relationships"]`, the task generates embeddings for the relationship (and for any missing subject/object nodes), creates or updates graph nodes, and adds the relationship edge to the knowledge graph. Progress and final status are stored in the task cache under the current task id. Individual relationship or node failures (including timeouts) are skipped so remaining items continue processing.

    Returns:
        str: The Celery task id for the ingestion run.

    Raises:
        Exception: Any unhandled exception is recorded to the task cache with status "failed" and then re-raised.
    """

    print(
        "[DEBUG (process_architect_relationships)]: Processing ",
        len(args.get("relationships", [])),
        " architect relationships",
    )

    relationships_data: List[dict] = args.get("relationships", [])
    brain_id: str = args.get("brain_id", "default")
    session_id: Optional[str] = args.get("session_id")

    try:
        cache_adapter.set(
            key=f"task:{self.request.id}",
            value=json.dumps(
                {
                    "status": "started",
                    "task_id": self.request.id,
                    "total_relationships": len(relationships_data),
                }
            ),
            brain_id=brain_id,
            expires_in=3600 * 24 * 7,
        )

        ingestion_manager = IngestionManager(
            embeddings_adapter, vector_store_adapter, graph_adapter
        )

        relationships = [
            ArchitectAgentRelationship(**rel_data) for rel_data in relationships_data
        ]

        with ThreadPoolExecutor(max_workers=10) as io_executor:
            rel_embedding_futures: List[Tuple[Future, ArchitectAgentRelationship]] = []
            for relationship in relationships:
                if not isinstance(relationship, ArchitectAgentRelationship):
                    print(
                        f"[!] Skipping invalid relationship type: {type(relationship)}"
                    )
                    continue
                future = io_executor.submit(
                    ingestion_manager.process_rel_vectors,
                    relationship,
                    brain_id,
                )
                rel_embedding_futures.append((future, relationship))

            for future, relationship in rel_embedding_futures:
                print(f"> Processing relationship {relationship.name}")
                try:
                    v_id, v_rel_id = future.result(timeout=180)

                    subject_exists = graph_adapter.check_node_existence(
                        uuid=relationship.tail.uuid,
                        name=relationship.tail.name,
                        labels=[relationship.tail.type],
                        brain_id=brain_id,
                    )
                    object_exists = graph_adapter.check_node_existence(
                        uuid=relationship.tip.uuid,
                        name=relationship.tip.name,
                        labels=[relationship.tip.type],
                        brain_id=brain_id,
                    )
                    similar_v_rels = []
                    if v_rel_id is not None:
                        rel_vectors = vector_store_adapter.get_by_ids(
                            [str(v_rel_id)],
                            store="relationships",
                            brain_id=brain_id,
                        )
                        if rel_vectors and getattr(rel_vectors[0], "embeddings", None):
                            similar_v_rels = vector_store_adapter.search_vectors(
                                rel_vectors[0].embeddings,
                                brain_id=brain_id,
                                store="relationships",
                                k=10,
                            )
                    if similar_v_rels and similar_v_rels[0].distance > 0.9:
                        similar_rel = graph_adapter.get_triples_by_uuid(
                            [similar_v_rels[0].metadata.get("uuid")],
                            brain_id=brain_id,
                        )
                        if similar_rel:
                            similar_tail, _, similar_tip = similar_rel[0]
                            if (
                                similar_tail.uuid == relationship.tail.uuid
                                and similar_tip.uuid == relationship.tip.uuid
                            ) or (
                                similar_tail.uuid == relationship.tip.uuid
                                and similar_tip.uuid == relationship.tail.uuid
                            ):
                                vector_store_adapter.remove_vectors(
                                    [v_rel_id],
                                    store="relationships",
                                    brain_id=brain_id,
                                )
                                continue

                    node_embedding_futures = []
                    print(f"> Subject exists: {subject_exists}")
                    print(f"> Object exists: {object_exists}")
                    if not subject_exists:
                        future = io_executor.submit(
                            ingestion_manager.process_node_vectors,
                            relationship.tail,
                            brain_id,
                        )
                        node_embedding_futures.append((future, relationship.tail))
                    if not object_exists:
                        future = io_executor.submit(
                            ingestion_manager.process_node_vectors,
                            relationship.tip,
                            brain_id,
                        )
                        node_embedding_futures.append((future, relationship.tip))

                    graph_nodes = []

                    for future, node_data in node_embedding_futures:
                        print(f"> Processing node {node_data.name}")
                        try:
                            future.result(timeout=180)
                            graph_nodes.append(
                                Node(
                                    uuid=node_data.uuid,
                                    labels=[node_data.type],
                                    name=node_data.name,
                                    description=node_data.description,
                                    properties=node_data.properties,
                                    polarity=(
                                        node_data.polarity
                                        if node_data.polarity
                                        else "neutral"
                                    ),
                                )
                            )
                        except FutureTimeoutError:
                            print(
                                f"[!] Node embedding future timed out for {node_data.name}, skipping"
                            )
                            continue
                        except Exception as e:
                            print(
                                f"[!] Node embedding future failed for {node_data.name}: {e}"
                            )
                            continue

                    graph_adapter.add_nodes(graph_nodes, brain_id=brain_id)
                    print(f"> Added {len(graph_nodes)} nodes")
                    graph_adapter.add_relationship(
                        Node(
                            uuid=relationship.tail.uuid,
                            labels=[relationship.tail.type],
                            name=relationship.tail.name,
                            polarity=(
                                relationship.tail.polarity
                                if relationship.tail.polarity
                                else "neutral"
                            ),
                            **(
                                {"happened_at": relationship.tail.happened_at}
                                if relationship.tail.happened_at
                                else {}
                            ),
                            properties={
                                **(relationship.tail.properties or {}),
                            },
                        ),
                        Predicate(
                            uuid=relationship.uuid,
                            flow_key=relationship.flow_key,
                            name=relationship.name,
                            description=relationship.description,
                            properties={
                                **(relationship.properties or {}),
                                "v_id": v_rel_id,
                            },
                            last_updated=datetime.datetime.now(),
                            amount=relationship.amount,
                        ),
                        Node(
                            uuid=relationship.tip.uuid,
                            labels=[relationship.tip.type],
                            name=relationship.tip.name,
                            polarity=(
                                relationship.tip.polarity
                                if relationship.tip.polarity
                                else "neutral"
                            ),
                            **(
                                {"happened_at": relationship.tip.happened_at}
                                if relationship.tip.happened_at
                                else {}
                            ),
                            properties={
                                **(relationship.tip.properties or {}),
                            },
                        ),
                        brain_id=brain_id,
                    )
                except FutureTimeoutError:
                    rel_name = getattr(relationship, "name", "unknown")
                    print(
                        f"[!] Relationship embedding future timed out for {rel_name}, skipping"
                    )
                    continue
                except Exception as e:
                    rel_name = getattr(relationship, "name", "unknown")
                    print(
                        f"[!] Relationship embedding future failed for {rel_name}: {e}"
                    )
                    continue

        cache_adapter.set(
            key=f"task:{self.request.id}",
            value=json.dumps({"status": "completed", "task_id": self.request.id}),
            brain_id=brain_id,
            expires_in=3600 * 24 * 7,
        )

        if session_id:
            from src.lib.redis.client import _redis_client

            remaining = _redis_client.client.decr(
                f"{brain_id}:session:{session_id}:pending_tasks"
            )
            print(
                f"[DEBUG (process_architect_relationships)]: Session {session_id} has {remaining} remaining tasks"
            )

        return self.request.id

    except Exception as e:
        error_result = {
            "status": "failed",
            "task_id": self.request.id,
            "error": str(e),
            "args": args,
        }

        cache_adapter.set(
            key=f"task:{self.request.id}",
            value=json.dumps(error_result),
            brain_id=brain_id,
            expires_in=3600 * 24 * 7,
        )

        if session_id:
            from src.lib.redis.client import _redis_client

            _redis_client.client.decr(f"{brain_id}:session:{session_id}:pending_tasks")

        raise


@ingestion_app.task(bind=True)
def ingest_structured_data(self, args: dict):
    """
    Ingest structured elements into the knowledge graph by creating nodes, embedding and storing vectors, generating observations, and saving structured-data records.

    Parses `args` into an IngestionStructuredRequestBody and for each element:
    - creates a graph node with properties and metadata,
    - records node property change logs (KG changes),
    - embeds and stores node and element vectors in the vector store,
    - generates and stores observations (vectors and Observation records),
    - merges and saves structured-data records,
    - calls knowledge-graph enrichment for the element,
    and updates a task status cache with "started" and "completed" entries.

    Parameters:
        args (dict): The raw task payload parsed into IngestionStructuredRequestBody (must include brain_id and data elements).

    Returns:
        str: The task id for this ingestion (self.request.id).

    Exceptions:
        On exception, stores a "failed" task status with error details in the cache and re-raises the exception.
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
            _element = element.model_dump(mode="json")
            description = llm_small_adapter.generate_text(
                prompt=NODE_DESCRIPTION_PROMPT.format(
                    element=json.dumps(
                        {
                            **_element.get("json_data", {}),
                            **_element.get("textual_data", {}),
                        },
                        indent=2,
                    ),
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
                            **(element.textual_data if element.textual_data else {}),
                        },
                    )
                ],
                brain_id=payload.brain_id,
                identification_params=element.identification_params.model_dump(
                    mode="json"
                ),
                metadata={
                    **(element.metadata if element.metadata else {}),
                },
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

            existing_structured_data = data_adapter.get_structured_data_by_id(
                id=uuid, brain_id=payload.brain_id
            )

            new_structured_data = None

            if existing_structured_data:
                new_structured_data = existing_structured_data.model_copy(
                    update={
                        "data": {
                            **(
                                existing_structured_data.data
                                if existing_structured_data.data
                                else {}
                            ),
                            **(element.json_data if element.json_data else {}),
                        },
                        "types": [
                            *existing_structured_data.types,
                            *element.types,
                        ],
                        "metadata": {
                            **(
                                existing_structured_data.metadata
                                if existing_structured_data.metadata
                                else {}
                            ),
                            **(element.metadata if element.metadata else {}),
                        },
                        "brain_version": BRAIN_VERSION,
                    },
                )
                # TODO: update the changelogs here
            else:
                vector = embeddings_adapter.embed_text(node.name)
                vector.id = node.uuid
                vector.metadata = {
                    "labels": node.labels,
                    "name": node.name,
                    "uuid": node.uuid,
                }
                vector_store_adapter.add_vectors(vectors=[vector], store="nodes")
                new_structured_data = StructuredData(
                    id=uuid,
                    data={
                        **(element.json_data if element.json_data else {}),
                        **(element.textual_data if element.textual_data else {}),
                    },
                    types=element.types,
                    metadata=element.metadata,
                    brain_version=BRAIN_VERSION,
                )

            data_adapter.save_structured_data(
                structured_data=new_structured_data,
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
                (
                    json.dumps(element.textual_data, indent=2)
                    if len(element.textual_data.keys()) > 0
                    else description
                ),
                targeting=node,
                brain_id=payload.brain_id,
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


@ingestion_app.task(bind=True)
def consolidate_graph_async(
    self,
    session_id: str,
    brain_id: str = "default",
    ingestion_session_id: str = None,
):
    """
    Consolidate graph after all processing tasks complete.
    """
    import os
    import langsmith
    from src.core.layers.graph_consolidation.graph_consolidation import (
        consolidate_graph,
    )
    from src.lib.redis.client import _redis_client
    from src.config import config

    print(
        f"[DEBUG (consolidate_graph_async)]: Starting consolidation for session {session_id}"
    )

    if not config.run_graph_consolidator:
        print(
            "[DEBUG (consolidate_graph_async)]: Graph consolidator is disabled, skipping"
        )
        return

    relationships_data_str = _redis_client.get(
        f"session:{session_id}:relationships", brain_id=brain_id
    )
    if not relationships_data_str:
        print(
            f"[DEBUG (consolidate_graph_async)]: No relationships data found for session {session_id}"
        )
        return

    relationships_data = json.loads(relationships_data_str)
    relationships = [
        ArchitectAgentRelationship(**rel_data) for rel_data in relationships_data
    ]

    print(
        f"[DEBUG (consolidate_graph_async)]: Consolidating graph with {len(relationships)} relationships"
    )

    project_name = os.getenv("LANGSMITH_PROJECT", "brainapi")
    tracing_metadata = {"brain_id": brain_id, "flow": "consolidate_graph"}
    if ingestion_session_id:
        tracing_metadata["ingestion_session_id"] = ingestion_session_id

    try:
        with langsmith.tracing_context(
            project_name=project_name,
            enabled=True,
            tags=["consolidate_graph", "janitor", "kg_agent"],
            metadata=tracing_metadata,
        ):
            consolidation_response = consolidate_graph(relationships, brain_id=brain_id)
        try:
            from langchain_core.tracers.langchain import wait_for_all_tracers

            wait_for_all_tracers()
        except ImportError:
            pass
        print(
            f"[DEBUG (consolidate_graph_async)]: Consolidation completed for session {session_id}"
        )
        _redis_client.delete(f"session:{session_id}:relationships", brain_id=brain_id)
        _redis_client.client.delete(f"{brain_id}:session:{session_id}:pending_tasks")
        return (
            consolidation_response.model_dump(mode="json")
            if hasattr(consolidation_response, "model_dump")
            else None
        )
    except Exception as e:
        print(
            f"[DEBUG (consolidate_graph_async)]: Consolidation failed for session {session_id}: {e}"
        )
        raise
