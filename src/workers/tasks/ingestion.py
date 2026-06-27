"""
File: /ingestion.py
Created Date: Sunday October 19th 2025
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Wednesday March 4th 2026 9:35:41 pm
Modified By: Christian Nonis <alch.infoemail@gmail.com>
-----
"""

import base64
import datetime
import json
import os
import tempfile
import time
import tomllib
from concurrent.futures import (
    Future,
    ThreadPoolExecutor,
)
from concurrent.futures import (
    TimeoutError as FutureTimeoutError,
)
from pathlib import Path
from typing import List, Optional, Tuple, TypedDict
from uuid import uuid4

from pydantic import BaseModel

from src.config import config
from src.constants.agents import ArchitectAgentRelationship
from src.constants.data import (
    KGChangeLogNodePropertiesUpdated,
    KGChangeLogPredicateUpdatedProperty,
    KGChanges,
    KGChangesType,
    Observation,
    PartialNode,
    StructuredData,
    TextChunk,
)
from src.constants.kg import IdentificationParams, Node, Predicate
from src.constants.prompts.misc import NODE_DESCRIPTION_PROMPT
from src.constants.tasks.ingestion import (
    IngestionTaskArgs,
    IngestionTaskDataType,
    IngestionTaskTextArgs,
)
from src.core.agents.architect_agent import ArchitectAgent
from src.core.agents.kg_agent import KGAgent
from src.core.agents.scout_agent import ScoutAgent, ScoutEntity
from src.core.plugins.prompts import prompt_registry
from src.core.saving.auto_kg import enrich_kg_from_input
from src.core.saving.ingestion_manager import IngestionManager
from src.services.api.constants.requests import (
    IngestionStructuredRequestBody,
    IngestionTripleSet,
    PartialNodeFilter,
)
from src.services.data.main import data_adapter
from src.services.input.agents import llm_small_adapter
from src.services.kg_agent.main import (
    cache_adapter,
    embeddings_adapter,
    graph_adapter,
    vector_store_adapter,
)
from src.services.observations.main import observations_agent
from src.utils.dates import normalize_date_string
from src.utils.similarity.vectors import cosine_similarity
from src.workers.app import ingestion_app

PYPROJECT_PATH = Path(__file__).parent.parent.parent.parent / "pyproject.toml"
with open(PYPROJECT_PATH, "rb") as f:
    BRAIN_VERSION = tomllib.load(f)["project"]["version"]

NODE_RESOLUTION_SIMILARITY = 0.9
EVENT_RESOLUTION_NAME_SIMILARITY = 0.7


def _entity_key(entity) -> Tuple[str, str]:
    return (
        (entity.name or "").strip().lower(),
        (entity.type or "").strip().lower(),
    )


def _is_same_graph_node(graph_node: Node, entity) -> bool:
    if graph_node.uuid and graph_node.uuid == entity.uuid:
        return True
    same_name = (graph_node.name or "").strip().lower() == (
        entity.name or ""
    ).strip().lower()
    labels = [str(label).strip().lower() for label in (graph_node.labels or [])]
    return same_name and (entity.type or "").strip().lower() in labels


def _normalize_relationship_dates(
    relationships: List[ArchitectAgentRelationship],
) -> None:
    for relationship in relationships:
        for entity in (relationship.tail, relationship.tip):
            if (entity.type or "").strip().upper() == "DATE":
                entity.name = normalize_date_string(entity.name)
            if getattr(entity, "happened_at", None):
                entity.happened_at = normalize_date_string(entity.happened_at)


def _resolve_relationship_entities(
    relationships: List[ArchitectAgentRelationship], brain_id: str
) -> None:
    unique_entities = {}
    for relationship in relationships:
        for entity in (relationship.tail, relationship.tip):
            unique_entities.setdefault(_entity_key(entity), entity)

    embeddings_cache: dict = {}

    def _name_embedding(name: str):
        key = (name or "").strip().lower()
        if key not in embeddings_cache:
            try:
                vector = embeddings_adapter.embed_text(name)
                embeddings_cache[key] = vector.embeddings if vector else None
            except Exception:
                embeddings_cache[key] = None
        return embeddings_cache[key]

    def _exact_match(entity) -> Optional[Node]:
        try:
            return graph_adapter.get_by_identification_params(
                IdentificationParams(name=entity.name, entity_types=[entity.type]),
                brain_id=brain_id,
                entity_types=[entity.type],
            )
        except Exception:
            return None

    def _vector_match(entity) -> Optional[Node]:
        embedding = _name_embedding(entity.name)
        if not embedding:
            return None
        try:
            candidates = vector_store_adapter.search_vectors(
                embedding, brain_id=brain_id, store="nodes", k=5
            )
        except Exception:
            return None
        entity_type = (entity.type or "").strip().lower()
        for candidate in candidates:
            metadata = candidate.metadata or {}
            candidate_uuid = metadata.get("uuid")
            labels = [
                str(label).strip().lower() for label in metadata.get("labels") or []
            ]
            if not candidate_uuid or entity_type not in labels:
                continue
            candidate_vectors = vector_store_adapter.get_by_ids(
                [candidate.id], store="nodes", brain_id=brain_id
            )
            if not candidate_vectors or not candidate_vectors[0].embeddings:
                continue
            similarity = cosine_similarity(embedding, candidate_vectors[0].embeddings)
            if similarity < NODE_RESOLUTION_SIMILARITY:
                continue
            node = graph_adapter.get_by_uuid(candidate_uuid, brain_id=brain_id)
            if node:
                return node
        return None

    resolutions: dict = {}
    for key, entity in unique_entities.items():
        if key[1] == "event":
            continue
        node = _exact_match(entity) or _vector_match(entity)
        if node:
            resolutions[key] = node

    def _apply_resolutions():
        for relationship in relationships:
            for entity in (relationship.tail, relationship.tip):
                node = resolutions.get(_entity_key(entity))
                if node:
                    entity.uuid = node.uuid
                    entity.name = node.name

    _apply_resolutions()
    resolved_uuids = {node.uuid for node in resolutions.values()}

    for key, event in unique_entities.items():
        if key[1] != "event" or key in resolutions:
            continue
        anchor_uuids = set()
        for relationship in relationships:
            if (
                _entity_key(relationship.tip) == key
                and relationship.tail.uuid in resolved_uuids
            ):
                anchor_uuids.add(relationship.tail.uuid)
            if (
                _entity_key(relationship.tail) == key
                and relationship.tip.uuid in resolved_uuids
            ):
                anchor_uuids.add(relationship.tip.uuid)
        if not anchor_uuids:
            continue
        try:
            neighbor_map = graph_adapter.get_neighbors(
                list(anchor_uuids), of_types=["EVENT"], brain_id=brain_id
            )
        except Exception:
            continue
        candidate_counts: dict = {}
        candidate_nodes: dict = {}
        for pairs in (neighbor_map or {}).values():
            seen = set()
            for _, neighbor in pairs:
                if not neighbor or not neighbor.uuid or neighbor.uuid in seen:
                    continue
                seen.add(neighbor.uuid)
                candidate_nodes[neighbor.uuid] = neighbor
                candidate_counts[neighbor.uuid] = (
                    candidate_counts.get(neighbor.uuid, 0) + 1
                )
        event_embedding = _name_embedding(event.name)
        if not event_embedding:
            continue
        event_date = normalize_date_string(getattr(event, "happened_at", None))
        required_anchors = max(1, (len(anchor_uuids) + 1) // 2)
        best = None
        for candidate_uuid, count in candidate_counts.items():
            if count < required_anchors:
                continue
            candidate = candidate_nodes[candidate_uuid]
            candidate_date = normalize_date_string(
                getattr(candidate, "happened_at", None)
                or (candidate.properties or {}).get("happened_at")
            )
            if event_date and candidate_date and event_date != candidate_date:
                continue
            candidate_embedding = _name_embedding(candidate.name)
            if not candidate_embedding:
                continue
            similarity = cosine_similarity(event_embedding, candidate_embedding)
            if similarity < EVENT_RESOLUTION_NAME_SIMILARITY:
                continue
            if best is None or similarity > best[0]:
                best = (similarity, candidate)
        if best:
            resolutions[key] = best[1]

    _apply_resolutions()


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
            brain_id=payload.brain_id,
        )

        if config.pipeline_mode == "lightweight":
            print("[DEBUG (ingest_data)]: Lightweight pipeline mode selected")

        if config.pipeline_mode == "accurate":
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
        _normalize_relationship_dates(relationships)
        _resolve_relationship_entities(relationships, brain_id)

        with ThreadPoolExecutor(max_workers=10) as io_executor:
            rel_embedding_futures: List[Tuple[Future, ArchitectAgentRelationship]] = []
            for relationship in relationships:
                if not isinstance(relationship, ArchitectAgentRelationship):
                    print(
                        f"[!] Skipping invalid relationship type: {type(relationship)}"
                    )
                    continue
                if relationship.tail.uuid == relationship.tip.uuid:
                    print(
                        f"[!] Skipping self-relationship {relationship.name} on {relationship.tail.name}"
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
                                _is_same_graph_node(similar_tail, relationship.tail)
                                and _is_same_graph_node(similar_tip, relationship.tip)
                            ) or (
                                _is_same_graph_node(similar_tail, relationship.tip)
                                and _is_same_graph_node(similar_tip, relationship.tail)
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
                                    properties={
                                        k: v
                                        for k, v in (
                                            node_data.properties or {}
                                        ).items()
                                        if v is not None
                                    },
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
                            description=relationship.description or "",
                            properties={
                                **{
                                    k: v
                                    for k, v in (relationship.properties or {}).items()
                                    if v is not None
                                },
                                **(
                                    {"v_id": v_rel_id}
                                    if v_rel_id is not None
                                    else {}
                                ),
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
    Ingest event-centric information triples into the knowledge graph enriching subgraphs and registering the information in the memory.

    Parses `args` into an IngestionStructuredRequestBody and for each element:
    -

    Parameters:
        args (dict): The raw task payload parsed into IngestionStructuredRequestBody.

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

        anchor = None
        ingestion_manager = IngestionManager(
            embeddings_adapter, vector_store_adapter, graph_adapter
        )

        if isinstance(payload.anchor, PartialNodeFilter) and payload.anchor.uuid:
            anchor = graph_adapter.get_by_uuid(
                payload.anchor.uuid, brain_id=payload.brain_id
            )
            if not anchor:
                cache_adapter.set(
                    key=f"task:{self.request.id}",
                    value=json.dumps(
                        {
                            "status": "failed",
                            "task_id": self.request.id,
                            "error": f"Anchor node with uuid {payload.anchor.uuid} not found",
                        }
                    ),
                    brain_id=payload.brain_id,
                    expires_in=3600 * 24 * 7,
                )
                return

        else:
            anchor = graph_adapter.get_by_identification_params(
                IdentificationParams(
                    name=payload.anchor.name, entity_types=[payload.anchor.type]
                )
            )
            if not anchor:
                txt = (
                    payload.anchor.name + "; " + (payload.anchor.meta_description or "")
                )
                embedded_anchor = embeddings_adapter.embed_text(txt)
                matching_vector_nodes = vector_store_adapter.search_vectors(
                    embedded_anchor.embeddings,
                    store="nodes",
                    brain_id=payload.brain_id,
                    k=10,
                )
                matching_nodes = graph_adapter.get_by_uuids(
                    [v.metadata.get("uuid") for v in matching_vector_nodes],
                    brain_id=payload.brain_id,
                )
                kg_agent = KGAgent(
                    llm_adapter=llm_small_adapter,
                    cache_adapter=cache_adapter,
                    kg=graph_adapter,
                    vector_store=vector_store_adapter,
                    embeddings=embeddings_adapter,
                    database_desc=graph_adapter.graphdb_description,
                )
                kg_agent_anchor_result = kg_agent.verify_entity_existence(
                    entity_name=payload.anchor.name,
                    entity_types=[payload.anchor.type],
                    entity_meta_description=payload.anchor.meta_description,
                    pool_nodes=matching_nodes,
                    brain_id=payload.brain_id,
                )
                if kg_agent_anchor_result.exists:
                    anchor = kg_agent_anchor_result.node
                elif payload.anchor:
                    anchor_entity = ScoutEntity(
                        name=payload.anchor.name,
                        type=payload.anchor.type,
                        description=payload.anchor.meta_description,
                    )
                    ingestion_manager.process_node_vectors(
                        anchor_entity, payload.brain_id
                    )
                    added_nodes = graph_adapter.add_nodes(
                        [
                            Node(
                                uuid=anchor_entity.uuid,
                                labels=[payload.anchor.type],
                                name=anchor_entity.name,
                                description=anchor_entity.description,
                                properties=anchor_entity.properties,
                            )
                        ],
                        brain_id=payload.brain_id,
                    )
                    anchor = added_nodes[0]

        if payload.text:
            current_triples: List[IngestionTripleSet] = []
            partial_triples: List[IngestionTripleSet] = []
            for triple in payload.data:
                if triple.subject and triple.subj_event:
                    current_triples.append(triple)
                else:
                    partial_triples.append(triple)

            scout_agent = ScoutAgent(
                llm_adapter=llm_small_adapter,
                cache_adapter=cache_adapter,
                kg=graph_adapter,
                vector_store=vector_store_adapter,
                embeddings=embeddings_adapter,
            )
            architect_agent = ArchitectAgent(
                llm_adapter=llm_small_adapter,
                cache_adapter=cache_adapter,
                kg=graph_adapter,
                vector_store=vector_store_adapter,
                embeddings=embeddings_adapter,
                ingestion_manager=ingestion_manager,
            )
            scout_agent_response = scout_agent.run_structured(
                text=payload.text if payload.text else None,
                brain_id=payload.brain_id,
                timeout=180,
                max_retries=3,
                ingestion_session_id=self.request.id,
                partial_triples=partial_triples,
                current_triples=current_triples,
            )
            architect_agent.run_structured(
                text=payload.text if payload.text else None,
                entities=scout_agent_response.entities,
                targeting=anchor,
                brain_id=payload.brain_id,
                timeout=180,
                max_retries=3,
                ingestion_session_id=self.request.id,
                partial_triples=partial_triples,
                current_triples=current_triples,
            )

        # TODO [SIR]: Add the triples to the knowledge graph + save the data + create the embeddings and save them -- save the text if present and find a way to save the rels (maybe isn't necessary?)

        # TODO [SIR]: Create the observations ??
        # NOTE [SIR]: structured is by definition ment to save precise information so the pipeline will only be granular/accurate
        # NOTE [SIR]: for the text ingestion the lightweight can be converted into ingestion of only observations instead whole text

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

    from src.config import config
    from src.core.layers.graph_consolidation.graph_consolidation import (
        consolidate_graph,
    )
    from src.lib.redis.client import _redis_client

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


FILE_INGEST_MAX_RETRIES = 3
FILE_INGEST_RETRY_DELAY = 0.1


@ingestion_app.task(bind=True)
def ingest_file(self, content_b64: str, filename: str, brain_id: str):
    """
    Ingest a file via Docling: convert to markdown (per page), enqueue one
    ingest_data task per page.
    """
    from celery.exceptions import OperationalError

    try:
        from docling.document_converter import DocumentConverter
    except ImportError as exc:
        raise RuntimeError(
            "OCR_MODE=docling requires the 'docling-ocr' optional dependency group. "
            "Install it with: `python scripts/install_extras.py docling-ocr` "
            "(or `make install-extras`). "
            "Alternatively set OCR_MODE=docparser in your .env to use the remote OCR pipeline."
        ) from exc

    cache_adapter.set(
        key=f"task:{self.request.id}",
        value=json.dumps({"status": "started", "task_id": self.request.id}),
        brain_id=brain_id,
        expires_in=3600 * 24 * 7,
    )
    content = base64.b64decode(content_b64)
    suffix = ""
    if filename:
        for ext in (".pdf", ".docx", ".pptx", ".html", ".htm"):
            if filename.lower().endswith(ext):
                suffix = ext
                break
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(content)
        tmp_path = tmp.name
    try:
        converter = DocumentConverter()
        result = converter.convert(tmp_path)
        num_pages = len(result.document.pages)
        if num_pages == 0:
            markdown = result.document.export_to_markdown()
            page_markdowns = [markdown] if markdown.strip() else []
        else:
            page_markdowns = [
                result.document.export_to_markdown(page_no=p)
                for p in range(1, num_pages + 1)
            ]
    finally:
        os.unlink(tmp_path)
    task_ids = [str(uuid4()) for _ in page_markdowns]
    for page_task_id in task_ids:
        cache_adapter.set(
            key=f"task:{page_task_id}",
            value=json.dumps({"status": "queued", "task_id": page_task_id}),
            brain_id=brain_id,
            expires_in=3600 * 24 * 7,
        )
    for page_task_id, markdown in zip(task_ids, page_markdowns):
        payload = {
            "data": IngestionTaskTextArgs(
                data_type="text", text_data=markdown
            ).model_dump(),
            "brain_id": brain_id,
        }
        for attempt in range(FILE_INGEST_MAX_RETRIES):
            try:
                ingest_data.apply_async(
                    args=[payload],
                    task_id=page_task_id,
                )
                break
            except OperationalError:
                if attempt == FILE_INGEST_MAX_RETRIES - 1:
                    raise
                time.sleep(FILE_INGEST_RETRY_DELAY * (attempt + 1))
    cache_adapter.set(
        key=f"task:{self.request.id}",
        value=json.dumps(
            {
                "status": "completed",
                "task_id": self.request.id,
                "task_ids": task_ids,
            }
        ),
        brain_id=brain_id,
        expires_in=3600 * 24 * 7,
    )
    return {"task_id": self.request.id, "task_ids": task_ids}
