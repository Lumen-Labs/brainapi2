"""
File: /auto_kg.py
Created Date: Sunday December 21st 2025
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Thursday February 19th 2026 7:45:12 pm
Modified By: Christian Nonis <alch.infoemail@gmail.com>
-----
"""

import json
import os
import uuid
from dataclasses import dataclass, field
from typing import List, Optional

from src.config import config
from src.constants.kg import Node
from src.core.agents.scout_agent import ScoutAgent, ScoutEntity
from src.core.agents.architect_agent import (
    ArchitectAgent,
)
from src.core.saving.architect_scratchpad import (
    build_scratchpad,
    format_unit_with_scratchpad,
    resolve_prior_context_mode,
    serialize_scratchpad,
)
from src.core.saving.ingestion_manager import IngestionManager
from src.core.saving.ingest_cost import IngestCostLedger, count_source_tokens, track_stage
from src.services.input.agents import (
    cache_adapter,
    embeddings_adapter,
    graph_adapter,
    llm_small_adapter,
    vector_store_adapter,
)
from src.utils.text_chunking import chunk_text, source_context, source_cut
from src.core.saving.ingest_limits import (
    IngestModelLimitError,
    invocation_limits,
    is_size_error,
    source_span_tokens,
    scout_payload_size,
)

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
        tags=["enrich_kg", "scout", "architect"],
        metadata={
            "ingestion_session_id": ingestion_session_id,
            "brain_id": brain_id,
            "flow": "enrich_kg_from_input",
            "source_chunk_id": source_chunk_id,
            "source_timestamp": source_timestamp,
        },
    ):
        token = invocation_limits.set((config.ingest_max_new_tokens or None,))
        try:
            return _enrich_kg_impl(
                input,
                targeting,
                brain_id,
                ingestion_session_id,
                source_chunk_id=source_chunk_id,
                source_timestamp=source_timestamp,
                preferred_extraction_entities=preferred_extraction_entities,
            )
        finally:
            invocation_limits.reset(token)


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
    ) -> None:
        if architect_mode in ("schema", "batch"):
            architect_agent.run_batch_extract(
                unit_text,
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
        else:
            architect_agent.run_tooler(
                unit_text,
                unit_entities,
                targeting=targeting,
                brain_id=brain_id,
                timeout=20000,
                ingestion_session_id=ingestion_session_id,
                mode=("coarse" if mode == "coarse" else "granular"),
                reset=reset,
            )
            cost_ledger.record_architect_unit(escalated=False)

    pending = []
    recent_entities = []

    def ingest_span(start: int, end: int, depth: int = 0) -> None:
        source = input[start:end]
        if not source.strip():
            return
        unit_text = source_context(input, start) + source
        stage = "scout"
        span_token = source_span_tokens.set(count_source_tokens(source)[0])
        try:
            with track_stage(cost_ledger, stage):
                response = scout_agent._run_chunk(
                    unit_text,
                    targeting=targeting,
                    brain_id=brain_id,
                    ingestion_session_id=ingestion_session_id,
                    mode=mode,
                    reference_time=source_timestamp,
                    preferred_extraction_entities=preferred_extraction_entities,
                )
            entities = list(response.entities)
            if not entities:
                return

            stage = "architect"
            if config.ingest_architect_per_unit:
                dense_parts = _dense_architect_parts(
                    source,
                    entities,
                    entity_threshold=config.ingest_architect_dense_entity_threshold,
                    dense_max_chars=config.ingest_architect_dense_max_chars,
                )
                if len(dense_parts) > 1:
                    raise IngestModelLimitError("Source span exceeds the dense entity policy")

            architect_text = unit_text
            if prior_context_mode == "scratchpad" and recent_entities:
                architect_text = _architect_unit_text_scratchpad(
                    source_context(input, start, include_excerpt=False) + source,
                    prior_entities=_merge_entities(recent_entities[-_PRIOR_UNIT_WINDOW:]),
                    prior_relationships=pending[-32:],
                    token_cap=scratchpad_token_cap,
                )
            distinct_entities = _merge_entities([entities])
            payload_tokens = count_source_tokens(
                json.dumps([entity.model_dump(mode="json", exclude={"properties"}) for entity in distinct_entities], ensure_ascii=False)
            )[0]
            payload_token = scout_payload_size.set((len(distinct_entities), payload_tokens))
            try:
                with track_stage(cost_ledger, stage):
                    _run_architect_unit(architect_text, entities, reset=True)
            finally:
                scout_payload_size.reset(payload_token)

            if mode == "granular" and config.ingest_defer_janitor:
                stage = "janitor"
                with track_stage(cost_ledger, stage):
                    architect_agent.run_batched_janitor(
                        text=unit_text,
                        brain_id=brain_id,
                        targeting=targeting,
                        batch_size=config.janitor_batch_size,
                        cost_ledger=cost_ledger,
                    )
            pending.extend(architect_agent.take_pending_relationships())
            recent_entities.append(entities)
            del recent_entities[:-_PRIOR_UNIT_WINDOW]
        except Exception as exc:
            if not is_size_error(exc):
                raise
            if len(source) <= 1 or depth >= 12:
                raise IngestModelLimitError(
                    f"Ingestion {stage} could not fit source span [{start}, {end})"
                ) from exc
            cut = start + source_cut(source, len(source) // 2)
            ingest_span(start, cut, depth + 1)
            ingest_span(cut, end, depth + 1)
        finally:
            source_span_tokens.reset(span_token)

    start = 0
    while start < len(input):
        # Keep the existing source-unit policy for deployments that enable it.
        size = 6000 if config.ingest_architect_per_unit else len(input)
        end = start + source_cut(input[start:], size)
        ingest_span(start, end)
        start = end

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
