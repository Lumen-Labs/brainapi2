"""
File: /auto_kg.py
Created Date: Sunday December 21st 2025
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Thursday February 19th 2026 7:45:12 pm
Modified By: Christian Nonis <alch.infoemail@gmail.com>
-----
"""

import os
import uuid
from dataclasses import dataclass, field
from functools import partial
from typing import Callable, List, Optional

from src.config import config
from src.constants.kg import Node
from src.core.agents.scout_agent import ScoutAgent, ScoutEntity
from src.core.agents.architect_agent import (
    ArchitectAgent,
    ArchitectOutputLimitError,
)
from src.core.saving.architect_scratchpad import (
    build_scratchpad,
    format_unit_with_scratchpad,
    resolve_prior_context_mode,
    serialize_scratchpad,
)
from src.core.saving.ingestion_manager import IngestionManager
from src.core.saving.ingest_cost import IngestCostLedger, track_stage
from src.services.input.agents import (
    cache_adapter,
    embeddings_adapter,
    graph_adapter,
    llm_small_adapter,
    vector_store_adapter,
)
from src.utils.text_chunking import chunk_text

import langsmith

_PRIOR_UNIT_WINDOW = 4


@dataclass
class EnrichmentOrchestrationResult:
    session_id: Optional[str]
    ingestion_session_id: str
    brain_id: str
    enrichment_relationships: List[dict] = field(default_factory=list)
    should_consolidate: bool = False
    cost: dict = field(default_factory=dict)


def _entity_merge_key(entity: ScoutEntity) -> tuple:
    return (
        (entity.name or "").strip().lower(),
        (entity.type or "").strip().lower(),
        getattr(entity, "happened_at", None) or "",
    )


def _merge_entities(chunks_entities: List[List[ScoutEntity]]) -> List[ScoutEntity]:
    merged: dict[tuple, ScoutEntity] = {}
    for entities in chunks_entities:
        for entity in entities:
            key = _entity_merge_key(entity)
            if key not in merged:
                merged[key] = entity
            else:
                existing = merged[key]
                if (entity.description or "") and len(entity.description or "") > len(
                    existing.description or ""
                ):
                    existing.description = entity.description
    return list(merged.values())


def _architect_unit_text(chunk: str, prior_chunks: List[str]) -> str:
    if not prior_chunks:
        return chunk
    prior = prior_chunks[-_PRIOR_UNIT_WINDOW:]
    preamble = "\n\n".join(
        f"[Prior context {i + 1}]\n{p}" for i, p in enumerate(prior)
    )
    return f"{preamble}\n\n[Current unit]\n{chunk}"


def _architect_unit_text_scratchpad(
    chunk: str,
    *,
    prior_entities: List[ScoutEntity],
    prior_relationships: list,
    token_cap: int,
) -> str:
    if not prior_entities and not prior_relationships:
        return chunk
    pad = build_scratchpad(prior_entities, prior_relationships)
    pad_text, _tokens = serialize_scratchpad(pad, token_cap=token_cap)
    return format_unit_with_scratchpad(chunk, pad_text)


def _entity_name(entity: ScoutEntity) -> str:
    return (getattr(entity, "name", None) or "").strip()


def _filter_entities_for_text(
    entities: List[ScoutEntity], text: str
) -> List[ScoutEntity]:
    """Keep entities whose names appear in the unit; fall back to all if none match."""
    if not entities or not text:
        return list(entities or [])
    hay = text.lower()
    matched = [e for e in entities if _entity_name(e) and _entity_name(e).lower() in hay]
    return matched or list(entities)


def _dense_architect_parts(
    chunk: str,
    entities: List[ScoutEntity],
    *,
    entity_threshold: int,
    dense_max_chars: int,
) -> List[str]:
    """
    Split entity-dense dialogue units into smaller schema extracts so the batch
    path stays populated instead of empty→tooler escalate storms.
    """
    cleaned = (chunk or "").strip()
    if not cleaned:
        return [cleaned]
    threshold = max(1, int(entity_threshold or 12))
    max_chars = max(200, int(dense_max_chars or 1200))
    if len(entities) < threshold:
        return [cleaned]
    if len(cleaned) <= max_chars and cleaned.count("\n") < 4:
        return [cleaned]
    parts = chunk_text(cleaned, max_chars=max_chars)
    return parts if parts else [cleaned]


def enrich_kg_from_input(
    input: str,
    targeting: Optional[Node] = None,
    brain_id: str = "default",
    source_chunk_id: Optional[str] = None,
    source_timestamp: Optional[str] = None,
    preferred_extraction_entities: Optional[List[str]] = None,
) -> EnrichmentOrchestrationResult:
    """
    Orchestrates enrichment of the knowledge graph from a free-text input.

    Runs the scout and architect agents to extract entities and relationships from the provided input
    and returns the collected persistence batch for the caller to write.
    """

    ingestion_session_id = str(uuid.uuid4())
    project_name = os.getenv("LANGSMITH_PROJECT", "brainapi")
    with langsmith.tracing_context(
        project_name=project_name,
        enabled=True,
        tags=["enrich_kg", "scout", "architect"],
        metadata={
            "ingestion_session_id": ingestion_session_id,
            "brain_id": brain_id,
            "flow": "enrich_kg_from_input",
            "source_chunk_id": source_chunk_id,
            "source_timestamp": source_timestamp,
        },
    ):
        return _enrich_kg_impl(
            input,
            targeting,
            brain_id,
            ingestion_session_id,
            source_chunk_id=source_chunk_id,
            source_timestamp=source_timestamp,
            preferred_extraction_entities=preferred_extraction_entities,
        )


def _enrich_kg_impl(
    input,
    targeting,
    brain_id: str,
    ingestion_session_id: str,
    source_chunk_id: Optional[str] = None,
    source_timestamp: Optional[str] = None,
    preferred_extraction_entities: Optional[List[str]] = None,
) -> EnrichmentOrchestrationResult:
    from src.core.saving.identity import stamp_provenance

    cost_ledger = IngestCostLedger()
    cost_ledger.set_source_text(input if isinstance(input, str) else str(input or ""))
    ingestion_manager = IngestionManager(
        embeddings_adapter, vector_store_adapter, graph_adapter
    )

    scout_agent = ScoutAgent(
        llm_small_adapter,
        cache_adapter,
        kg=graph_adapter,
        vector_store=vector_store_adapter,
        embeddings=embeddings_adapter,
    )
    architect_agent = ArchitectAgent(
        llm_small_adapter,
        cache_adapter,
        kg=graph_adapter,
        vector_store=vector_store_adapter,
        embeddings=embeddings_adapter,
        ingestion_manager=ingestion_manager,
    )

    mode = "coarse" if config.pipeline_mode == "lightweight" else "granular"
    architect_mode = config.ingest_architect_mode
    prior_context_mode = resolve_prior_context_mode(
        config.ingest_architect_prior_context,
        architect_mode,
    )
    scratchpad_token_cap = int(config.ingest_architect_scratchpad_token_cap or 500)
    print(
        f"[DEBUG (enrich_kg_from_input)]: {config.pipeline_mode} pipeline mode selected; "
        f"architect_mode={architect_mode}; prior_context={prior_context_mode}"
    )
    architect_agent.defer_janitor = bool(config.ingest_defer_janitor)

    def _run_architect_unit(
        unit_text: str,
        unit_entities: List[ScoutEntity],
        *,
        reset: bool,
        with_context: Optional[Callable[[str], str]] = None,
    ) -> None:
        if architect_mode in ("schema", "batch"):
            try:
                architect_agent.run_batch_extract(
                    with_context(unit_text) if with_context else unit_text,
                    unit_entities,
                    targeting=targeting,
                    brain_id=brain_id,
                    timeout=20000,
                    ingestion_session_id=ingestion_session_id,
                    mode=("coarse" if mode == "coarse" else "granular"),
                    reset=reset,
                    cost_ledger=cost_ledger,
                    escalate=bool(config.ingest_architect_escalate),
                )
            except ArchitectOutputLimitError:
                cleaned = unit_text.strip()
                if len(cleaned) <= 256:
                    raise
                parts = chunk_text(cleaned, max_chars=len(cleaned) // 2)
                for index, part in enumerate(parts):
                    _run_architect_unit(
                        part,
                        _filter_entities_for_text(unit_entities, part),
                        reset=reset and index == 0,
                        with_context=with_context,
                    )
        else:
            architect_agent.run_tooler(
                with_context(unit_text) if with_context else unit_text,
                unit_entities,
                targeting=targeting,
                brain_id=brain_id,
                timeout=20000,
                ingestion_session_id=ingestion_session_id,
                mode=("coarse" if mode == "coarse" else "granular"),
                reset=reset,
            )
            cost_ledger.record_architect_unit(escalated=False)

    if config.ingest_architect_per_unit:
        chunks = chunk_text(input, max_chars=6000)
        per_chunk_entities: List[List[ScoutEntity]] = []
        with track_stage(cost_ledger, "scout"):
            for chunk in chunks:
                response = scout_agent._run_chunk(
                    chunk,
                    targeting=targeting,
                    brain_id=brain_id,
                    ingestion_session_id=ingestion_session_id,
                    mode=("coarse" if mode == "coarse" else "granular"),
                    reference_time=source_timestamp,
                    preferred_extraction_entities=preferred_extraction_entities,
                )
                per_chunk_entities.append(list(response.entities))

        entities = _merge_entities(per_chunk_entities)
        print("[DEBUG (initial_scout_entities)]: ", entities)

        with track_stage(cost_ledger, "architect"):
            first_unit = True
            for idx, chunk in enumerate(chunks):
                chunk_entities = (
                    per_chunk_entities[idx] if idx < len(per_chunk_entities) else []
                )
                if not chunk_entities and not entities:
                    continue
                unit_entities = chunk_entities if chunk_entities else entities
                architect_parts = _dense_architect_parts(
                    chunk,
                    unit_entities,
                    entity_threshold=config.ingest_architect_dense_entity_threshold,
                    dense_max_chars=config.ingest_architect_dense_max_chars,
                )
                if len(architect_parts) > 1:
                    print(
                        "[DEBUG (enrich_kg_from_input)]: dense architect rechunk "
                        f"entities={len(unit_entities)} parts={len(architect_parts)} "
                        f"max_chars={config.ingest_architect_dense_max_chars}"
                    )
                for part in architect_parts:
                    part_entities = _filter_entities_for_text(unit_entities, part)
                    with_context = None
                    if prior_context_mode == "scratchpad" and not first_unit:
                        window_start = max(0, idx - _PRIOR_UNIT_WINDOW)
                        prior_ents: List[ScoutEntity] = []
                        for prior_idx in range(window_start, idx):
                            prior_ents.extend(per_chunk_entities[prior_idx])
                        with_context = partial(
                            _architect_unit_text_scratchpad,
                            prior_entities=prior_ents,
                            prior_relationships=list(
                                getattr(architect_agent, "relationships_set", []) or []
                            ),
                            token_cap=scratchpad_token_cap,
                        )
                    elif prior_context_mode == "raw" and not first_unit:
                        with_context = partial(_architect_unit_text, prior_chunks=chunks[:idx])
                    _run_architect_unit(
                        part, part_entities, reset=first_unit, with_context=with_context
                    )
                    first_unit = False
    else:
        with track_stage(cost_ledger, "scout"):
            scout_response = scout_agent.run(
                input,
                targeting=targeting,
                brain_id=brain_id,
                ingestion_session_id=ingestion_session_id,
                mode=("coarse" if mode == "coarse" else "granular"),
                reference_time=source_timestamp,
                preferred_extraction_entities=preferred_extraction_entities,
            )
        entities = list(scout_response.entities)
        print("[DEBUG (initial_scout_entities)]: ", entities)
        with track_stage(cost_ledger, "architect"):
            _run_architect_unit(input, entities, reset=True)

    if mode == "granular" and config.ingest_defer_janitor:
        with track_stage(cost_ledger, "janitor"):
            architect_agent.run_batched_janitor(
                text=input,
                brain_id=brain_id,
                targeting=targeting,
                batch_size=config.janitor_batch_size,
                cost_ledger=cost_ledger,
            )

    pending = architect_agent.take_pending_relationships()
    enrichment_relationships = []
    for rel in pending:
        payload = rel.model_dump(mode="json")
        payload["properties"] = stamp_provenance(
            payload.get("properties"),
            source_chunk_id=source_chunk_id,
            source_timestamp=source_timestamp,
        )
        for endpoint_key in ("tail", "tip"):
            endpoint = payload.get(endpoint_key) or {}
            endpoint["properties"] = stamp_provenance(
                endpoint.get("properties"),
                source_chunk_id=source_chunk_id,
                source_timestamp=source_timestamp,
            )
            payload[endpoint_key] = endpoint
        enrichment_relationships.append(payload)
    should_consolidate = bool(
        config.pipeline_mode == "accurate"
        and config.run_graph_consolidator
        and enrichment_relationships
    )

    try:
        from langchain_core.tracers.langchain import wait_for_all_tracers

        wait_for_all_tracers()
    except ImportError:
        pass

    return EnrichmentOrchestrationResult(
        session_id=architect_agent.session_id,
        ingestion_session_id=ingestion_session_id,
        brain_id=brain_id,
        enrichment_relationships=enrichment_relationships,
        should_consolidate=should_consolidate,
        cost=cost_ledger.to_dict(),
    )
