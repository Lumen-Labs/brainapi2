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

from src.core.saving.identity import (
    stable_node_id,
    stable_relationship_id,
    stamp_provenance,
)

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
RELATIONSHIP_DEDUP_MAX_DISTANCE = 0.1

TASK_STATUS_RANK = {
    "queued": 0,
    "started": 1,
    "persisting": 2,
    "consolidating": 3,
    "completed": 4,
    "partial_failed": 4,
    "failed": 4,
}
TERMINAL_TASK_STATUSES = {"completed", "failed", "partial_failed"}


def set_ingestion_task_status(
    task_id: str,
    brain_id: str,
    status: str,
    *,
    stage: Optional[str] = None,
    error: Optional[str] = None,
    errors: Optional[list] = None,
    counts: Optional[dict] = None,
    cost: Optional[dict] = None,
) -> dict:
    from src.core.saving.ingest_cost import IngestCostLedger, merge_cost_into_status_payload

    key = f"task:{task_id}"
    existing_raw = cache_adapter.get(key, brain_id=brain_id)
    existing: dict = {}
    if existing_raw:
        try:
            if isinstance(existing_raw, bytes):
                existing_raw = existing_raw.decode("utf-8")
            existing = json.loads(existing_raw) if isinstance(existing_raw, str) else dict(existing_raw)
        except Exception:
            existing = {}
    previous = existing.get("status")
    if previous in TERMINAL_TASK_STATUSES and status not in TERMINAL_TASK_STATUSES:
        return existing
    if previous and TASK_STATUS_RANK.get(status, -1) < TASK_STATUS_RANK.get(previous, -1):
        return existing
    payload = {
        **existing,
        "status": status,
        "task_id": task_id,
    }
    if stage is not None:
        payload["stage"] = stage
    if counts is not None:
        payload["counts"] = counts
    if error is not None:
        payload["error"] = str(error)[:2000]
    if errors is not None:
        payload["errors"] = [
            {
                "message": str(item.get("message", item))[:500],
                **(
                    {"relationship": item.get("relationship")}
                    if isinstance(item, dict) and item.get("relationship")
                    else {}
                ),
            }
            for item in errors[:50]
        ]
    if cost is not None:
        payload = merge_cost_into_status_payload(
            payload, IngestCostLedger.from_dict(cost)
        )
    cache_adapter.set(
        key=key,
        value=json.dumps(payload),
        brain_id=brain_id,
        expires_in=3600 * 24 * 7,
    )
    return payload


def nearest_existing_vector(candidates, exclude_id) -> Optional[object]:
    for candidate in candidates or []:
        if exclude_id is not None and str(candidate.id) == str(exclude_id):
            continue
        return candidate
    return None


def should_dedup_relationship(candidate, max_distance: float = RELATIONSHIP_DEDUP_MAX_DISTANCE) -> bool:
    return (
        candidate is not None
        and candidate.distance is not None
        and float(candidate.distance) < max_distance
    )


def _entity_key(entity) -> Tuple[str, str, str]:
    entity_uuid = (getattr(entity, "uuid", None) or "").strip()
    return (
        entity_uuid,
        (entity.name or "").strip().lower(),
        (entity.type or "").strip().lower(),
    )


def _is_same_graph_node(graph_node: Node, entity) -> bool:
    if graph_node.uuid and graph_node.uuid == getattr(entity, "uuid", None):
        return True
    same_name = (graph_node.name or "").strip().lower() == (
        entity.name or ""
    ).strip().lower()
    labels = [str(label).strip().lower() for label in (graph_node.labels or [])]
    return same_name and (entity.type or "").strip().lower() in labels


def _normalize_relationship_dates(
    relationships: List[ArchitectAgentRelationship],
    reference_time: Optional[str] = None,
) -> None:
    from src.utils.dates import resolve_relative_date

    for relationship in relationships:
        for entity in (relationship.tail, relationship.tip):
            if (entity.type or "").strip().upper() == "DATE":
                entity.name = resolve_relative_date(entity.name, reference_time)
            if getattr(entity, "happened_at", None):
                entity.happened_at = resolve_relative_date(
                    entity.happened_at, reference_time
                )


def _resolve_relationship_entities(
    relationships: List[ArchitectAgentRelationship],
    brain_id: str,
    source_chunk_id: Optional[str] = None,
    source_timestamp: Optional[str] = None,
) -> None:
    unique_entities = {}
    for relationship in relationships:
        for entity in (relationship.tail, relationship.tip):
            unique_entities.setdefault(_entity_key(entity), entity)

    pre_stable_rel_ids = {
        id(relationship): stable_relationship_id(
            relationship.tail.uuid,
            relationship.name,
            relationship.tip.uuid,
            relationship.flow_key,
        )
        for relationship in relationships
    }

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

    def _uuid_match(entity) -> Optional[Node]:
        entity_uuid = (getattr(entity, "uuid", None) or "").strip()
        if not entity_uuid:
            return None
        try:
            return graph_adapter.get_by_uuid(entity_uuid, brain_id=brain_id)
        except Exception:
            return None

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
        ranked = []
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
                ranked.append((similarity, node))
        if not ranked:
            return None
        ranked.sort(key=lambda item: item[0], reverse=True)
        if len(ranked) >= 2 and abs(ranked[0][0] - ranked[1][0]) < 0.02:
            pool_nodes = [node for _, node in ranked[:5]]
            try:
                from src.core.agents.kg_agent import KGAgent

                kg_agent = KGAgent(
                    llm_adapter=llm_small_adapter,
                    cache_adapter=cache_adapter,
                    kg=graph_adapter,
                    vector_store=vector_store_adapter,
                    embeddings=embeddings_adapter,
                    database_desc=graph_adapter.graphdb_description,
                )
                adjudication = kg_agent.verify_entity_existence(
                    entity_name=entity.name,
                    entity_types=[entity.type] if entity.type else [],
                    entity_meta_description=getattr(entity, "description", None),
                    pool_nodes=pool_nodes,
                    brain_id=brain_id,
                )
                if getattr(adjudication, "exists", False) and getattr(
                    adjudication, "node", None
                ):
                    return adjudication.node
            except Exception as exc:
                print(f"[!] Entity adjudication failed for {entity.name}: {exc}")
            return None
        return ranked[0][1]

    resolutions: dict = {}
    for key, entity in unique_entities.items():
        uuid_hit = _uuid_match(entity)
        if uuid_hit:
            resolutions[key] = uuid_hit
            continue
        if key[2] == "event":
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
                    if getattr(node, "happened_at", None) and not entity.happened_at:
                        entity.happened_at = node.happened_at
                    existing_props = getattr(node, "properties", None) or {}
                    entity.properties = stamp_provenance(
                        entity.properties,
                        source_chunk_id=source_chunk_id,
                        source_timestamp=source_timestamp,
                        existing_properties=existing_props,
                    )
                    if getattr(node, "description", None):
                        entity.description = _merge_description(
                            node.description, entity.description
                        )
                        aliases = list(existing_props.get("aliases") or [])
                        if entity.name and entity.name not in aliases and entity.name != node.name:
                            aliases.append(entity.name)
                        if aliases:
                            entity.properties = {
                                **(entity.properties or {}),
                                "aliases": aliases,
                            }

    _apply_resolutions()
    resolved_uuids = {node.uuid for node in resolutions.values()}

    for key, event in unique_entities.items():
        if key[2] != "event" or key in resolutions:
            continue
        uuid_hit = _uuid_match(event)
        if uuid_hit:
            resolutions[key] = uuid_hit
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
            if best is not None and abs(similarity - best[0]) < 0.02:
                best = None
                break
            if best is None or similarity > best[0]:
                best = (similarity, candidate)
        if best:
            resolutions[key] = best[1]

    _apply_resolutions()

    for relationship in relationships:
        relationship.properties = stamp_provenance(
            relationship.properties,
            source_chunk_id=source_chunk_id,
            source_timestamp=source_timestamp,
        )
        if source_timestamp and not (relationship.properties or {}).get("valid_at"):
            relationship.properties = {
                **(relationship.properties or {}),
                "valid_at": source_timestamp,
            }
        for entity in (relationship.tail, relationship.tip):
            entity.properties = stamp_provenance(
                entity.properties,
                source_chunk_id=source_chunk_id,
                source_timestamp=source_timestamp,
            )
            if resolutions.get(_entity_key(entity)):
                continue
            entity.uuid = stable_node_id(
                entity.name,
                entity.type,
                getattr(entity, "happened_at", None),
                getattr(entity, "uuid", None),
            )
        current_uuid = getattr(relationship, "uuid", None)
        pre_stable = pre_stable_rel_ids.get(id(relationship))
        if current_uuid and pre_stable and current_uuid != pre_stable:
            continue
        relationship.uuid = stable_relationship_id(
            relationship.tail.uuid,
            relationship.name,
            relationship.tip.uuid,
            relationship.flow_key,
        )


def _merge_description(
    existing: Optional[str], incoming: Optional[str]
) -> Optional[str]:
    existing_clean = (existing or "").strip()
    incoming_clean = (incoming or "").strip()
    if not existing_clean:
        return incoming_clean or None
    if not incoming_clean:
        return existing_clean
    if incoming_clean.lower() in existing_clean.lower():
        return existing_clean
    if existing_clean.lower() in incoming_clean.lower():
        return incoming_clean
    return f"{existing_clean} | {incoming_clean}"


def _is_event_entity(entity) -> bool:
    if (getattr(entity, "type", None) or "").strip().lower() == "event":
        return True
    return any(
        (str(label) or "").strip().lower() == "event"
        for label in (getattr(entity, "labels", None) or [])
    )


def _invalidate_superseded_relationships(
    relationship: ArchitectAgentRelationship,
    *,
    brain_id: str,
) -> None:
    """Mark older same-type outgoing edges from the same subject as invalid when tip changes.

    Event hub legs are never invalidated: an actor accumulates one leg per event and
    none of them supersede the others.
    """
    if _is_event_entity(relationship.tail) or _is_event_entity(relationship.tip):
        return
    if not (relationship.tail.type or "").strip() or not (
        relationship.tip.type or ""
    ).strip():
        return
    try:
        neighbors = graph_adapter.get_neighbors(
            [relationship.tail.uuid], brain_id=brain_id
        )
    except Exception:
        return
    pairs = (neighbors or {}).get(relationship.tail.uuid) or []
    valid_at = (relationship.properties or {}).get("valid_at") or (
        relationship.properties or {}
    ).get("source_timestamp")
    for predicate, neighbor in pairs:
        if not predicate or not neighbor:
            continue
        if (getattr(predicate, "direction", None) or "").strip().lower() != "out":
            continue
        if _is_event_entity(neighbor):
            continue
        if (predicate.name or "").strip().upper() != (relationship.name or "").strip().upper():
            continue
        if neighbor.uuid == relationship.tip.uuid:
            continue
        if getattr(predicate, "uuid", None) == relationship.uuid:
            continue
        props = getattr(predicate, "properties", None) or {}
        if props.get("invalid_at"):
            continue
        try:
            graph_adapter.update_properties(
                predicate.uuid,
                "relationship",
                brain_id=brain_id,
                new_properties={
                    "invalid_at": valid_at
                    or datetime.datetime.utcnow().strftime("%d/%m/%Y"),
                    "deprecated": True,
                },
            )
        except Exception as exc:
            print(f"[!] Failed to invalidate relationship {predicate.uuid}: {exc}")

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


def source_text_from_payload(payload: IngestionTaskArgs) -> str:
    if payload.data.data_type == IngestionTaskDataType.TEXT.value:
        return payload.data.text_data
    return json.dumps(payload.data.json_data)


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
        from src.config import validate_pipeline_mode

        validate_pipeline_mode(config.pipeline_mode)

        set_ingestion_task_status(
            self.request.id,
            payload.brain_id,
            "started",
            stage="source",
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

        source_text = source_text_from_payload(payload)

        text_chunk = data_adapter.save_text_chunk(
            TextChunk(
                text=source_text,
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

        from src.core.saving.ingest_cost import IngestCostLedger, estimate_tokens, track_stage

        cost_ledger = IngestCostLedger()
        with track_stage(cost_ledger, "embed"):
            cost_ledger.embed.add_usage(calls=1)

        if payload.skip_enrichment:
            set_ingestion_task_status(
                self.request.id,
                payload.brain_id,
                "completed",
                stage="completed",
                cost=cost_ledger.to_dict(),
            )
            return self.request.id

        if config.pipeline_mode == "accurate" and config.run_observations:
            with track_stage(cost_ledger, "observations"):
                observations = observations_agent.observe(
                    text=source_text,
                    observate_for=payload.observate_for,
                    context=None,
                )
                cost_ledger.observations.add_usage(
                    input_tokens=estimate_tokens(source_text),
                    output_tokens=estimate_tokens("\n".join(observations or [])),
                    calls=1,
                )

            data_adapter.save_observations(
                [
                    Observation(
                        text=observation,
                        metadata=payload.meta_keys,
                        resource_id=text_chunk.id,
                    )
                    for observation in observations
                ],
                brain_id=payload.brain_id,
            )

        enrich_result = enrich_kg_from_input(
            source_text,
            brain_id=payload.brain_id,
            source_chunk_id=text_chunk.id,
            source_timestamp=payload.source_timestamp,
            preferred_extraction_entities=payload.preferred_extraction_entities,
        )
        cost_ledger.merge(
            IngestCostLedger.from_dict(getattr(enrich_result, "cost", None))
        )
        parent_task_id = self.request.id
        set_ingestion_task_status(
            parent_task_id,
            payload.brain_id,
            "started",
            stage="enriched",
            cost=cost_ledger.to_dict(),
        )
        steps = []
        if enrich_result.enrichment_relationships:
            set_ingestion_task_status(
                parent_task_id,
                payload.brain_id,
                "persisting",
                stage="persisting",
                counts={
                    "relationships": len(enrich_result.enrichment_relationships),
                },
                cost=cost_ledger.to_dict(),
            )
            steps.append(
                process_architect_relationships.si(
                    {
                        "relationships": enrich_result.enrichment_relationships,
                        "brain_id": enrich_result.brain_id,
                        "session_id": enrich_result.session_id,
                        "parent_task_id": parent_task_id,
                    }
                )
            )
        if enrich_result.should_consolidate:
            set_ingestion_task_status(
                parent_task_id,
                payload.brain_id,
                "consolidating",
                stage="consolidating",
                cost=cost_ledger.to_dict(),
            )
            steps.append(
                consolidate_graph_async.si(
                    enrich_result.session_id,
                    enrich_result.brain_id,
                    enrich_result.ingestion_session_id,
                    enrich_result.enrichment_relationships,
                )
            )
        steps.append(
            finalize_ingestion_task.si(
                parent_task_id,
                payload.brain_id,
                "completed",
                None,
                None,
                None,
                cost_ledger.to_dict(),
            )
        )
        from celery import chain

        chain(*steps).apply_async()
        return parent_task_id

    except Exception as e:
        brain_id = payload.brain_id if payload else args.get("brain_id", "default")
        set_ingestion_task_status(
            self.request.id,
            brain_id,
            "failed",
            stage="failed",
            error=str(e),
        )
        raise


@ingestion_app.task(bind=True)
def finalize_ingestion_task(
    self,
    parent_task_id: str,
    brain_id: str = "default",
    status: str = "completed",
    error: Optional[str] = None,
    errors: Optional[list] = None,
    counts: Optional[dict] = None,
    cost: Optional[dict] = None,
):
    try:
        graph_adapter.rebuild_topic_index(brain_id)
    except Exception as e:
        print(f"[!] Topic index rebuild failed: {e}")
    set_ingestion_task_status(
        parent_task_id,
        brain_id,
        status,
        stage=status,
        error=error,
        errors=errors,
        counts=counts,
        cost=cost,
    )
    return parent_task_id


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
    parent_task_id: Optional[str] = args.get("parent_task_id")
    status_task_id = parent_task_id or self.request.id

    try:
        set_ingestion_task_status(
            status_task_id,
            brain_id,
            "persisting",
            stage="persisting",
            counts={"relationships": len(relationships_data)},
        )

        ingestion_manager = IngestionManager(
            embeddings_adapter, vector_store_adapter, graph_adapter
        )

        relationships = [
            ArchitectAgentRelationship(**rel_data) for rel_data in relationships_data
        ]
        reference_time = None
        source_chunk_id = None
        for rel in relationships:
            props = rel.properties or {}
            if not reference_time:
                reference_time = props.get("source_timestamp") or props.get("valid_at")
            if not source_chunk_id:
                ids = props.get("source_chunk_ids") or []
                if ids:
                    source_chunk_id = ids[-1]
                elif props.get("source_chunk_id"):
                    source_chunk_id = props.get("source_chunk_id")
        _normalize_relationship_dates(relationships, reference_time=reference_time)
        _resolve_relationship_entities(
            relationships,
            brain_id,
            source_chunk_id=source_chunk_id,
            source_timestamp=reference_time,
        )

        item_errors: List[dict] = []
        with ThreadPoolExecutor(max_workers=10) as io_executor:
            rel_embedding_futures: List[Tuple[Future, ArchitectAgentRelationship]] = []
            for relationship in relationships:
                if not isinstance(relationship, ArchitectAgentRelationship):
                    print(
                        f"[!] Skipping invalid relationship type: {type(relationship)}"
                    )
                    item_errors.append(
                        {
                            "relationship": str(type(relationship)),
                            "message": "invalid relationship type",
                        }
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
                    nearest_existing = nearest_existing_vector(
                        similar_v_rels, v_rel_id
                    )
                    if should_dedup_relationship(nearest_existing):
                        similar_rel = graph_adapter.get_triples_by_uuid(
                            [nearest_existing.metadata.get("uuid")],
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
                            props = {
                                k: v
                                for k, v in (node_data.properties or {}).items()
                                if v is not None
                            }
                            labels = [node_data.type]
                            extra = props.get("catalog_labels") or []
                            if isinstance(extra, str):
                                extra = [extra]
                            for item in extra:
                                label = str(item).strip()
                                if label and label not in labels:
                                    labels.append(label)
                            graph_nodes.append(
                                Node(
                                    uuid=node_data.uuid,
                                    labels=labels,
                                    name=node_data.name,
                                    description=node_data.description,
                                    properties=props,
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
                    _invalidate_superseded_relationships(
                        relationship,
                        brain_id=brain_id,
                    )
                except FutureTimeoutError:
                    rel_name = getattr(relationship, "name", "unknown")
                    item_errors.append(
                        {
                            "relationship": rel_name,
                            "message": "relationship embedding timed out",
                        }
                    )
                except Exception as e:
                    rel_name = getattr(relationship, "name", "unknown")
                    item_errors.append(
                        {"relationship": rel_name, "message": str(e)}
                    )

        if item_errors:
            status = (
                "partial_failed"
                if len(item_errors) < len(relationships_data)
                else "failed"
            )
            print(
                f"[!] Relationship persistence failures ({len(item_errors)}): "
                f"{item_errors}"
            )
            set_ingestion_task_status(
                status_task_id,
                brain_id,
                status,
                stage="persisting",
                errors=item_errors,
                counts={
                    "relationships": len(relationships_data),
                    "failed": len(item_errors),
                },
            )
            raise RuntimeError(
                f"{len(item_errors)} relationship persistence failures"
            )

        try:
            import time as _time

            touched_entities: set[str] = set()
            for relationship in relationships:
                for endpoint in (relationship.tail, relationship.tip):
                    typ = (getattr(endpoint, "type", None) or "").strip().upper()
                    if typ and typ != "EVENT" and getattr(endpoint, "uuid", None):
                        touched_entities.add(str(endpoint.uuid))
            started = _time.perf_counter()
            if touched_entities:
                graph_adapter.refresh_hub_bridges_for_entities(
                    list(touched_entities), brain_id=brain_id
                )
            else:
                graph_adapter.rebuild_hub_bridge_index(brain_id)
            index_ms = (_time.perf_counter() - started) * 1000.0
            if parent_task_id:
                from src.core.saving.ingest_cost import IngestCostLedger

                ledger = IngestCostLedger()
                ledger.index_rebuild.add_usage(latency_ms=index_ms, calls=1)
                set_ingestion_task_status(
                    parent_task_id,
                    brain_id,
                    "persisting",
                    stage="indexing",
                    cost=ledger.to_dict(),
                )
        except Exception as e:
            print(f"[!] Hub bridge index rebuild failed: {e}")

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
        current = cache_adapter.get(f"task:{status_task_id}", brain_id=brain_id)
        current_status = None
        if current:
            try:
                parsed = json.loads(
                    current.decode("utf-8") if isinstance(current, bytes) else current
                )
                current_status = parsed.get("status")
            except Exception:
                current_status = None
        if current_status not in TERMINAL_TASK_STATUSES:
            set_ingestion_task_status(
                status_task_id,
                brain_id,
                "failed",
                stage="persisting",
                error=str(e),
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
        mode = payload.resolved_mode()

        set_ingestion_task_status(
            self.request.id,
            payload.brain_id,
            "started",
            stage="source",
            counts={"triples": len(payload.data), "mode": mode},
        )

        if mode == "deterministic" and payload.anchor is not None:
            if not payload.anchor.uuid and not (
                payload.anchor.name and payload.anchor.type
            ):
                error = (
                    "deterministic mode requires anchor.uuid or exact name+type; "
                    "LLM entity verification is not allowed"
                )
                set_ingestion_task_status(
                    self.request.id,
                    payload.brain_id,
                    "failed",
                    stage="anchor",
                    error=error,
                )
                raise ValueError(error)

        anchor = None
        ingestion_manager = IngestionManager(
            embeddings_adapter, vector_store_adapter, graph_adapter
        )

        if payload.anchor is None:
            pass
        elif payload.anchor.uuid:
            anchor = graph_adapter.get_by_uuid(
                payload.anchor.uuid, brain_id=payload.brain_id
            )
            if not anchor:
                error = f"Anchor node with uuid {payload.anchor.uuid} not found"
                set_ingestion_task_status(
                    self.request.id,
                    payload.brain_id,
                    "failed",
                    stage="anchor",
                    error=error,
                )
                raise ValueError(error)
        else:
            anchor = graph_adapter.get_by_identification_params(
                IdentificationParams(
                    name=payload.anchor.name, entity_types=[payload.anchor.type]
                ),
                brain_id=payload.brain_id,
                entity_types=[payload.anchor.type] if payload.anchor.type else None,
            )
            if not anchor:
                if mode == "deterministic":
                    error = (
                        "deterministic mode: anchor not found by exact name+type; "
                        "provide anchor.uuid or create the node first "
                        "(LLM verify_entity_existence is disabled)"
                    )
                    set_ingestion_task_status(
                        self.request.id,
                        payload.brain_id,
                        "failed",
                        stage="anchor",
                        error=error,
                    )
                    raise ValueError(error)
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
                else:
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

        data_adapter.save_structured_data(
            StructuredData(
                id=self.request.id,
                data={
                    "triples": [t.model_dump(mode="json") for t in payload.data],
                    "text": payload.text,
                    "mode": mode,
                    "anchor": (
                        payload.anchor.model_dump(mode="json")
                        if payload.anchor
                        else None
                    ),
                },
                types=["ingestion_structured"],
                metadata={"task_id": self.request.id, "mode": mode},
                brain_version=BRAIN_VERSION,
            ),
            brain_id=payload.brain_id,
        )

        current_triples: List[IngestionTripleSet] = []
        partial_triples: List[IngestionTripleSet] = []
        for triple in payload.data:
            if triple.subject and triple.subj_event:
                current_triples.append(triple)
            else:
                partial_triples.append(triple)

        from src.core.agents.architect_agent import ingestion_triples_to_relationships

        required_relationships, _ = ingestion_triples_to_relationships(
            current_triples, partial_triples
        )
        for rel in required_relationships:
            props = dict(rel.properties or {})
            props.setdefault("source", "structured_deterministic")
            rel.properties = props
        persistence_batches = [
            rel.model_dump(mode="json") for rel in required_relationships
        ]
        session_id = None

        run_llm_enrichment = mode in ("hybrid", "enrich") and bool(payload.text)
        if mode == "deterministic":
            run_llm_enrichment = False

        if run_llm_enrichment:
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
                text=payload.text,
                brain_id=payload.brain_id,
                timeout=180,
                max_retries=3,
                ingestion_session_id=self.request.id,
                partial_triples=partial_triples,
                current_triples=current_triples,
            )
            architect_agent.run_structured(
                text=payload.text,
                entities=scout_agent_response.entities,
                targeting=anchor,
                brain_id=payload.brain_id,
                timeout=180,
                max_retries=3,
                ingestion_session_id=self.request.id,
                partial_triples=partial_triples,
                current_triples=current_triples,
                persist_submitted=False,
            )
            enrichment = architect_agent.take_pending_relationships()
            session_id = architect_agent.session_id
            for rel in enrichment:
                props = dict(rel.properties or {})
                props["source"] = "llm_enrichment"
                rel.properties = props
            persistence_batches.extend(
                rel.model_dump(mode="json") for rel in enrichment
            )

        parent_task_id = self.request.id
        steps = []
        if persistence_batches:
            set_ingestion_task_status(
                parent_task_id,
                payload.brain_id,
                "persisting",
                stage="persisting",
                counts={"relationships": len(persistence_batches)},
            )
            steps.append(
                process_architect_relationships.si(
                    {
                        "relationships": persistence_batches,
                        "brain_id": payload.brain_id,
                        "session_id": session_id,
                        "parent_task_id": parent_task_id,
                    }
                )
            )
        steps.append(
            finalize_ingestion_task.si(
                parent_task_id,
                payload.brain_id,
                "completed",
            )
        )
        from celery import chain

        chain(*steps).apply_async()
        return parent_task_id

    except Exception as e:
        brain_id = payload.brain_id if payload else args.get("brain_id", "default")
        set_ingestion_task_status(
            self.request.id,
            brain_id,
            "failed",
            stage="failed",
            error=str(e),
        )
        raise


@ingestion_app.task(bind=True)
def consolidate_graph_async(
    self,
    session_id: str,
    brain_id: str = "default",
    ingestion_session_id: str = None,
    relationships: Optional[List[dict]] = None,
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

    relationships_data = relationships
    if not relationships_data:
        relationships_data_str = _redis_client.get(
            f"session:{session_id}:relationships", brain_id=brain_id
        )
        if relationships_data_str:
            relationships_data = json.loads(relationships_data_str)
    if not relationships_data:
        print(
            f"[DEBUG (consolidate_graph_async)]: No relationships data found for session {session_id}"
        )
        return

    relationship_models = [
        ArchitectAgentRelationship(**rel_data) for rel_data in relationships_data
    ]

    print(
        f"[DEBUG (consolidate_graph_async)]: Consolidating graph with {len(relationship_models)} relationships"
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
            consolidate_graph(relationship_models, brain_id=brain_id)
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
        return None
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


def build_url_page_ingest_payload(
    brain_id: str,
    page,
    skip_enrichment: bool = False,
) -> dict:
    meta_keys = {}
    if getattr(page, "url", None):
        meta_keys["source_url"] = page.url
    if getattr(page, "title", None):
        meta_keys["source_title"] = page.title
    if getattr(page, "content_hash", None):
        meta_keys["content_hash"] = page.content_hash
    if getattr(page, "page_type", None):
        meta_keys["page_type"] = page.page_type
    payload = {
        "data": IngestionTaskTextArgs(
            data_type="text", text_data=page.text
        ).model_dump(),
        "brain_id": brain_id,
        "skip_enrichment": skip_enrichment,
    }
    if meta_keys:
        payload["meta_keys"] = meta_keys
    return payload


@ingestion_app.task(bind=True)
def ingest_url(self, args: dict):
    from celery.exceptions import OperationalError

    from src.adapters.interfaces.crawler import CrawlRequest
    from src.core.instances import crawler_adapter
    from src.lib.crawler.errors import (
        CrawlerConfigError,
        CrawlerFailedError,
        CrawlerQuotaError,
        CrawlerTimeoutError,
    )

    brain_id = args.get("brain_id", "default")
    skip_enrichment = bool(args.get("skip_enrichment", False))
    try:
        set_ingestion_task_status(
            self.request.id,
            brain_id,
            "started",
            stage="crawling",
        )
        pages = crawler_adapter.crawl_pages(
            CrawlRequest(
                url=args["url"],
                max_pages=int(args.get("max_pages") or 10),
                allowed_domains=args.get("allowed_domains") or [],
                idempotency_key=self.request.id,
            )
        )
        ingestible = [page for page in pages if (page.text or "").strip()]
        task_ids = [str(uuid4()) for _ in ingestible]
        for page_task_id in task_ids:
            set_ingestion_task_status(
                page_task_id,
                brain_id,
                "queued",
                stage="queued",
            )
        for page_task_id, page in zip(task_ids, ingestible):
            payload = build_url_page_ingest_payload(
                brain_id,
                page,
                skip_enrichment=skip_enrichment,
            )
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
        set_ingestion_task_status(
            self.request.id,
            brain_id,
            "completed",
            counts={"pages": len(task_ids), "task_ids": task_ids},
        )
        return {"task_id": self.request.id, "task_ids": task_ids}
    except (
        CrawlerConfigError,
        CrawlerQuotaError,
        CrawlerFailedError,
        CrawlerTimeoutError,
    ) as exc:
        set_ingestion_task_status(
            self.request.id,
            brain_id,
            "failed",
            stage="failed",
            error=str(exc),
        )
        raise
    except Exception as exc:
        set_ingestion_task_status(
            self.request.id,
            brain_id,
            "failed",
            stage="failed",
            error=str(exc),
        )
        raise
