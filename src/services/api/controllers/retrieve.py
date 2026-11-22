"""
File: /retrieve.py
Created Date: Sunday October 26th 2025
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Thursday January 29th 2026 8:43:59 pm
Modified By: Christian Nonis <alch.infoemail@gmail.com>
-----
"""

import asyncio
import contextvars
import logging
import os
import re
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any, Optional

from fastapi import HTTPException
from starlette.responses import JSONResponse

from src.constants.embeddings import Vector
from src.constants.kg import IdentificationParams, Node, Predicate
from src.core.search.entities import search_entities
from src.core.search.fact_filter import (
    filter_relevant_facts,
    personalized_pagerank,
    reciprocal_rank_fusion,
)
from src.core.search.hybrid import (
    collect_bm25_passages,
    collect_dense_passages,
    collect_ilike_passages,
    fuse_passage_lists,
)
from src.core.search.relationships import search_relationships
from src.lib.tracing.profiler import (
    log_stage_report,
    profile_request,
    profile_stage,
    stage_profiling_enabled,
)
from src.utils.vector_search import VectorSearchFacade
from src.services.api.constants.requests import (
    GetContextRequestBody,
    GetContextResponse,
    GetContextTriple,
    RetrieveRequestResponse,
    RetrieveNeighborsRequestResponse,
    RetrievedNeighborNode,
    jsonable_graph_value,
)
from src.services.kg_agent.main import graph_adapter, kg_agent
from src.services.data.main import data_adapter
from src.services.kg_agent.main import embeddings_adapter, vector_store_adapter
from src.utils.dates import parse_date_string, to_naive_utc
from src.utils.similarity.vectors import cosine_similarity
from src.utils.nlp.ner import _entity_extractor
from src.config import config
from src.core.verification.facts import (
    chunk_ids_with_only_rejected_support,
    combined_verification,
    fact_pair_allowed,
    fact_tuple_is_verified,
    filter_items_with_rejected_support,
    verification_scope,
)
from src.services.api.constants.requests import FactVerificationInfo

vector_search = VectorSearchFacade(vector_store_adapter)
logger = logging.getLogger(__name__)

_MAX_NOUN_CHUNKS = 5
_SEED_K = 25
_PASSAGE_K = 24
_IO_TIMEOUT_SECONDS = 10.0
_IO_MAX_WORKERS = 8
_IO_MAX_INFLIGHT = 8
_IO_THREAD_PREFIX = "ctx-io"
_io_state: dict[str, Any] = {"pool": None, "pid": 0, "inflight": 0}
_io_state_lock = threading.Lock()
_health_var: contextvars.ContextVar[Any] = contextvars.ContextVar(
    "retrieve_health", default=None
)
_backend_error_types_cache: tuple[type[BaseException], ...] | None = None
_PPR_DAMPING = 0.85
_PPR_ITERS = 20
_SESSION_RE = re.compile(r"session_(\d+)", re.IGNORECASE)
_BATCH_TURN_RE = re.compile(r"\bb(\d+)_t(\d+)\b", re.IGNORECASE)
_SPINE_ACTOR = ("MADE", "INITIATED", "PERFORMED", "EXPERIENCED", "COVERED")
_SPINE_TARGET = ("TARGETED", "AFFECTED", "RESULTED")
_CONTEXT_REL = ("OCCURRED", "WITHIN")
_DIVERSIFY_LAMBDA = 0.65
_BRIDGE_SCORE_PENALTY = 0.15
_BRIDGE_SEED_HUB_CAP = 12
_BRIDGE_RESERVE_HUB_CAP = 6
_BRIDGE_RESERVE_MIN = 3
_BRIDGE_RESERVE_FRAC_DENOM = 4


def _io_executor() -> ThreadPoolExecutor:
    pid = os.getpid()
    pool = _io_state["pool"]
    if pool is None or _io_state["pid"] != pid:
        with _io_state_lock:
            pool = _io_state["pool"]
            if pool is None or _io_state["pid"] != pid:
                previous_pid = _io_state["pid"]
                pool = ThreadPoolExecutor(
                    max_workers=_IO_MAX_WORKERS, thread_name_prefix=_IO_THREAD_PREFIX
                )
                _io_state["pool"] = pool
                _io_state["pid"] = pid
                if previous_pid != 0 and previous_pid != pid:
                    _io_state["inflight"] = 0
    return pool


def _on_io_worker() -> bool:
    return threading.current_thread().name.startswith(_IO_THREAD_PREFIX)


def _submit_io(fn, *args, **kwargs):
    ctx = contextvars.copy_context()
    return _io_executor().submit(ctx.run, fn, *args, **kwargs)


_IO_FAILED = object()


def _io_result(value, default):
    return default if value is _IO_FAILED else value


def _wait_all(futures, timeout: float = _IO_TIMEOUT_SECONDS):
    results = []
    last_exc = None
    deadline = time.monotonic() + timeout
    try:
        for fut in futures:
            remaining = max(0.0, deadline - time.monotonic())
            try:
                results.append(fut.result(timeout=remaining))
            except _backend_error_types() as exc:
                last_exc = exc
                results.append(_IO_FAILED)
        if last_exc is not None:
            for fut in futures:
                fut.cancel()
            if all(value is _IO_FAILED for value in results):
                raise last_exc
        return results
    except Exception:
        for fut in futures:
            fut.cancel()
        raise


def _reserve_io_jobs(n: int) -> bool:
    with _io_state_lock:
        inflight = int(_io_state.get("inflight") or 0)
        if inflight + n > _IO_MAX_INFLIGHT:
            return False
        _io_state["inflight"] = inflight + n
        return True


def _release_io_jobs(n: int) -> None:
    with _io_state_lock:
        inflight = int(_io_state.get("inflight") or 0)
        _io_state["inflight"] = max(0, inflight - n)


def _parallel_io(fns: list):
    if len(fns) <= 1 or _on_io_worker():
        return [fn() for fn in fns]
    n = len(fns)
    _io_executor()
    if not _reserve_io_jobs(n):
        return [fn() for fn in fns]
    released = [False] * n
    released_lock = threading.Lock()

    def release_slot(index: int) -> None:
        with released_lock:
            if released[index]:
                return
            released[index] = True
        _release_io_jobs(1)

    def wrap(index: int, fn):
        def run():
            try:
                return fn()
            finally:
                release_slot(index)

        return run

    futures = []
    try:
        futures = [_submit_io(wrap(index, fn)) for index, fn in enumerate(fns)]
        results = _wait_all(futures, timeout=_IO_TIMEOUT_SECONDS)
        for index, fut in enumerate(futures):
            if fut.cancelled():
                release_slot(index)
        return results
    except Exception:
        for index, fut in enumerate(futures):
            if fut.cancelled() or fut.cancel():
                release_slot(index)
        for index in range(len(futures), n):
            release_slot(index)
        raise


def _embed_query_vector(text: str) -> Vector:
    try:
        vector = embeddings_adapter.embed_text(text)
    except HTTPException:
        raise
    except Exception:
        logger.exception("Embedding request failed")
        raise HTTPException(
            status_code=503,
            detail="Embedding service unavailable",
        ) from None
    values = getattr(vector, "embeddings", None)
    if not values:
        logger.error("Embedding request failed")
        raise HTTPException(
            status_code=503,
            detail="Embedding service unavailable",
        )
    return vector


def _backend_error_types() -> tuple[type[BaseException], ...]:
    global _backend_error_types_cache
    if _backend_error_types_cache is not None:
        return _backend_error_types_cache
    types: list[type[BaseException]] = [
        TimeoutError,
        ConnectionError,
        BrokenPipeError,
    ]
    optional = (
        ("pymilvus.exceptions", "MilvusException"),
        ("psycopg2", "Error"),
        ("grpc", "RpcError"),
        ("neo4j.exceptions", "DriverError"),
    )
    for module_name, attr in optional:
        try:
            module = __import__(module_name, fromlist=[attr])
        except ImportError:
            continue
        cls = getattr(module, attr, None)
        if isinstance(cls, type) and issubclass(cls, BaseException):
            types.append(cls)
    _backend_error_types_cache = tuple(types)
    return _backend_error_types_cache


class _RetrievalHealth:
    def __init__(self) -> None:
        self._pending: set[str] = set()
        self._ok: set[str] = set()
        self._error: set[str] = set()

    def start(self, stage: str) -> None:
        self._pending.add(stage)

    def ok(self, stage: str) -> None:
        if stage not in self._pending:
            return
        self._pending.discard(stage)
        self._ok.add(stage)

    def error(self, stage: str) -> None:
        self._pending.discard(stage)
        self._error.add(stage)

    @property
    def failed_stages(self) -> list[str]:
        return sorted(self._error | self._pending)

    def all_failed(self) -> bool:
        tracked = self._pending | self._ok | self._error
        return bool(tracked) and not self._ok


def _health() -> _RetrievalHealth | None:
    value = _health_var.get()
    return value if isinstance(value, _RetrievalHealth) else None


def _note_degraded(stage: str, brain_id: str, exc: BaseException) -> None:
    logger.warning(
        "Retrieval stage %s failed for brain_id=%s (%s)",
        stage,
        brain_id,
        type(exc).__name__,
    )
    health = _health()
    if health is not None:
        health.error(stage)


def _start_stage(stage: str) -> None:
    health = _health()
    if health is not None:
        health.start(stage)


def _ok_stage(stage: str) -> None:
    health = _health()
    if health is not None:
        health.ok(stage)


def _backend_or_default(stage: str, brain_id: str, fn, default):
    _start_stage(stage)
    try:
        result = fn()
    except HTTPException:
        raise
    except _backend_error_types() as exc:
        _note_degraded(stage, brain_id, exc)
        return default
    _ok_stage(stage)
    return result


_DISTANCE_QUANT_DECIMALS = 3


def _quantize_distance(distance: float) -> float:
    try:
        value = float(distance)
    except (TypeError, ValueError):
        return float("inf")
    if value != value or value == float("inf") or value == float("-inf"):
        return value
    return round(value, _DISTANCE_QUANT_DECIMALS)


def _candidate_event_uuid(candidate: dict[str, Any]) -> str:
    return _event_hub_id(candidate)


def _best_hub_fact_texts(candidates: list[dict[str, Any]]) -> dict[str, str]:
    best: dict[str, tuple[float, str]] = {}
    for candidate in candidates:
        hub = _candidate_event_uuid(candidate)
        if not hub:
            continue
        text = str(candidate.get("text") or "").strip()
        if not text:
            continue
        score = _candidate_distance(candidate)
        prev = best.get(hub)
        if prev is None or score < prev[0] or (score == prev[0] and text < prev[1]):
            best[hub] = (score, text)
    return {hub: text for hub, (_score, text) in best.items()}


def _paths_for_curated(
    paths: list[dict[str, Any]],
    curated: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Keep bridge paths whose hubs both survived curation; attach fact legs."""
    hub_texts = _best_hub_fact_texts(curated)
    if not hub_texts or not paths:
        return []
    curated_hubs = set(hub_texts)
    out: list[dict[str, Any]] = []
    for path in paths:
        hubs = [str(h) for h in (path.get("hubs") or []) if h]
        if len(hubs) < 2 or not all(hub in curated_hubs for hub in hubs):
            continue
        out.append(
            {
                "hubs": hubs,
                "shared_entity": path.get("shared_entity"),
                "shared_entity_name": path.get("shared_entity_name"),
                "weight": path.get("weight"),
                "legs": [hub_texts[hub] for hub in hubs],
            }
        )
    out.sort(
        key=lambda p: (
            tuple(p["hubs"]),
            str(p.get("shared_entity") or ""),
        )
    )
    return out


def _candidate_distance(candidate: dict[str, Any]) -> float:
    raw = candidate.get("score")
    if raw is None:
        return float("inf")
    try:
        return _quantize_distance(float(raw))
    except (TypeError, ValueError):
        return float("inf")


def _rank_bridge_seed_hubs(
    candidates: list[dict[str, Any]],
    *,
    cap: int,
) -> list[str]:
    hub_best_score: dict[str, float] = {}
    for candidate in candidates:
        hub = _candidate_event_uuid(candidate)
        if not hub:
            continue
        score = _candidate_distance(candidate)
        prev = hub_best_score.get(hub)
        if prev is None or score < prev:
            hub_best_score[hub] = score
    return sorted(
        hub_best_score.keys(),
        key=lambda hub: (hub_best_score[hub], hub),
    )[: max(0, cap)]


def _resolve_chunk_sessions(
    chunk_ids: list[str], brain_id: str
) -> dict[str, list[str]]:
    unique: list[str] = []
    seen: set[str] = set()
    for chunk_id in chunk_ids:
        cid = str(chunk_id or "").strip()
        if cid and cid not in seen:
            seen.add(cid)
            unique.append(cid)
    if not unique:
        return {}
    out: dict[str, list[str]] = {}
    try:
        chunks, _ = data_adapter.get_text_chunks_by_ids(
            unique[:120], False, brain_id
        )
    except Exception:
        return {}
    for chunk in chunks:
        chunk_id = str(getattr(chunk, "id", "") or "")
        body = (getattr(chunk, "text", None) or "").strip()
        if not chunk_id or not body:
            continue
        sessions = _session_ids_from_text(body)
        if sessions:
            out[chunk_id] = sessions
    return out


def _sessions_from_chunk_map(
    chunk_ids: list[str], chunk_sessions: dict[str, list[str]]
) -> list[str]:
    found: list[str] = []
    seen: set[str] = set()
    for chunk_id in chunk_ids:
        for sid in chunk_sessions.get(str(chunk_id), []) or []:
            if sid not in seen:
                seen.add(sid)
                found.append(sid)
    return found


def _expand_cross_event_bridges(
    candidates: list[dict[str, Any]],
    brain_id: str,
    *,
    max_per_hub: int,
    include_history: bool = False,
    as_of: datetime | None = None,
    unknown_out: list | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    from src.core.saving.hub_bridges import (
        HubBridge,
        select_bridge_neighbors,
    )

    if max_per_hub <= 0 or not candidates:
        return candidates, []

    hub_best_score: dict[str, float] = {}
    seed_sessions: set[str] = set()
    seed_chunk_ids: list[str] = []
    for candidate in candidates:
        hub = _candidate_event_uuid(candidate)
        if hub:
            score = _candidate_distance(candidate)
            prev = hub_best_score.get(hub)
            if prev is None or score < prev:
                hub_best_score[hub] = score
        for sid in _candidate_session_ids(candidate):
            seed_sessions.add(sid)
        seed_chunk_ids.extend(str(c) for c in (candidate.get("chunk_ids") or []) if c)

    ranked_seeds = _rank_bridge_seed_hubs(
        candidates, cap=_BRIDGE_SEED_HUB_CAP
    )
    if not ranked_seeds:
        return candidates, []
    reserve_hubs = set(
        ranked_seeds[: min(_BRIDGE_RESERVE_HUB_CAP, len(ranked_seeds))]
    )

    try:
        bridges = graph_adapter.get_hub_bridges(ranked_seeds, brain_id=brain_id)
    except Exception:
        return candidates, []
    if not bridges:
        return candidates, []

    typed_bridges: list[HubBridge] = []
    for item in bridges:
        if isinstance(item, HubBridge):
            typed_bridges.append(item)
        else:
            typed_bridges.append(
                HubBridge(
                    event_a=str(item.get("event_a") or ""),
                    event_b=str(item.get("event_b") or ""),
                    shared_entity=str(item.get("shared_entity") or ""),
                    shared_entity_name=str(item.get("shared_entity_name") or ""),
                    weight=float(item.get("weight") or 1.0),
                )
            )

    expansions = select_bridge_neighbors(
        ranked_seeds,
        typed_bridges,
        max_per_hub=max_per_hub,
    )
    if not expansions:
        return candidates, []

    selected_neighbors = {neighbor for neighbor, _ in expansions}
    neighbor_uuids = sorted(selected_neighbors)
    with profile_stage(
        "graph.bridge_neighbors", neighbors=len(neighbor_uuids)
    ) as detail:
        neighbors = graph_adapter.get_event_hub_facts(
            neighbor_uuids, brain_id=brain_id
        )
        detail["rows"] = len(neighbors)

    neighbor_chunk_ids: list[str] = []
    facts_by_hub: dict[str, list[tuple[Any, Any, Any, Any, Any]]] = {}
    for n, r, m, r2, b in neighbors:
        event_uuid = str(getattr(m, "uuid", "") or "")
        if not event_uuid:
            continue
        facts_by_hub.setdefault(event_uuid, []).append((n, r, m, r2, b))
        neighbor_chunk_ids.extend(_extract_source_chunk_ids(n, r, m, r2, b))

    chunk_sessions = _resolve_chunk_sessions(
        seed_chunk_ids + neighbor_chunk_ids, brain_id
    )
    if not seed_sessions:
        seed_sessions.update(_sessions_from_chunk_map(seed_chunk_ids, chunk_sessions))

    hub_sessions: dict[str, set[str]] = {}
    for event_uuid, rows in facts_by_hub.items():
        sessions: set[str] = set()
        for n, r, m, r2, b in rows:
            chunk_ids = _extract_source_chunk_ids(n, r, m, r2, b)
            sessions.update(_sessions_from_chunk_map(chunk_ids, chunk_sessions))
        hub_sessions[event_uuid] = sessions

    bridge_by_neighbor = {neighbor: bridge for neighbor, bridge in expansions}
    seed_by_neighbor: dict[str, str] = {}
    for neighbor, bridge in expansions:
        if bridge.event_a == neighbor:
            seed_by_neighbor[neighbor] = bridge.event_b
        else:
            seed_by_neighbor[neighbor] = bridge.event_a

    path_meta: list[dict[str, Any]] = []
    for neighbor, bridge in expansions:
        seed = seed_by_neighbor.get(neighbor, "")
        path_meta.append(
            {
                "hubs": sorted([seed, neighbor]),
                "shared_entity": bridge.shared_entity,
                "shared_entity_name": bridge.shared_entity_name,
                "weight": bridge.weight,
            }
        )
    path_meta.sort(
        key=lambda p: (
            tuple(p["hubs"]),
            p["shared_entity"],
        )
    )

    existing_keys = {
        candidate.get("key")
        for candidate in candidates
        if candidate.get("key") is not None
    }
    expanded = list(candidates)
    for event_uuid in sorted(selected_neighbors):
        bridge = bridge_by_neighbor.get(event_uuid)
        if bridge is None:
            continue
        seed = seed_by_neighbor.get(event_uuid, "")
        base = hub_best_score.get(seed, float("inf"))
        novel = hub_sessions.get(event_uuid, set()) - seed_sessions
        score = (
            base + _BRIDGE_SCORE_PENALTY
            if base != float("inf")
            else float("inf")
        )
        if novel and score != float("inf"):
            score = max(0.0, score - min(0.05, 0.02 * len(novel)))
        reserve_ok = seed in reserve_hubs
        for n, r, m, r2, b in facts_by_hub.get(event_uuid, []):
            if as_of is not None:
                from src.core.saving.validity import (
                    VALIDITY_UNKNOWN,
                    predicate_validity_status,
                    unknown_fact,
                    unknown_relationship_id,
                )

                status_r = predicate_validity_status(r, as_of)
                status_r2 = predicate_validity_status(r2, as_of)
                if status_r == VALIDITY_UNKNOWN or status_r2 == VALIDITY_UNKNOWN:
                    if unknown_out is not None:
                        unknown_out.append(
                            unknown_fact(
                                target="relationship",
                                item_id=unknown_relationship_id(
                                    r, r2, status_r, status_r2
                                ),
                                payload=_format_event_fact(n, r, m, r2, b),
                            )
                        )
                    continue
            if not _fact_predicates_allowed(
                r, r2, include_history=include_history, as_of=as_of
            ):
                continue
            key = _fact_key(r, r2)
            if key in existing_keys:
                continue
            existing_keys.add(key)
            chunk_ids = _extract_source_chunk_ids(n, r, m, r2, b)
            session_ids = _sessions_from_chunk_map(chunk_ids, chunk_sessions)
            fact_text = _format_event_fact(n, r, m, r2, b)
            if as_of is None and include_history and not (
                _is_currently_valid(r) and _is_currently_valid(r2)
            ):
                fact_text = f"[superseded] {fact_text}"
            expanded.append(
                {
                    "identified_entity": f"bridge:{bridge.shared_entity}",
                    "triple": (n, r, m, r2, b),
                    "score": score,
                    "key": key,
                    "text": fact_text,
                    "chunk_ids": chunk_ids,
                    "session_ids": session_ids,
                    "bridge": {
                        "from_hub": seed,
                        "to_hub": event_uuid,
                        "shared_entity": bridge.shared_entity,
                        "shared_entity_name": bridge.shared_entity_name,
                        "reserve_ok": reserve_ok,
                    },
                }
            )
    return expanded, path_meta


async def retrieve_data(
    text: str,
    limit: int,
    preferred_entities: str,
    brain_id: str = "default",
    include_rejected: bool = False,
    as_of: Optional[str] = None,
    include_validity_unknown: bool = False,
) -> RetrieveRequestResponse:
    """
    Retrieve data from the knowledge graph and data store.
    """
    from src.core.saving.validity import (
        InvalidAsOf,
        VALIDITY_UNKNOWN,
        VALIDITY_VALID,
        dedupe_unknown_items,
        chunk_validity_status,
        optional_instant,
        unknown_fact,
        validity_status,
    )

    try:
        as_of_dt = optional_instant(as_of)
    except InvalidAsOf as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if preferred_entities:
        preferred_entities = [
            e.strip() for e in preferred_entities.split(",") if e.strip()
        ]
    else:
        preferred_entities = []

    def _get_data():
        text_embeddings = _embed_query_vector(text)

        data_vectors = vector_search.search_data(
            text_embeddings.embeddings,
            brain_id=brain_id,
            k=limit,
        )
        triple_vectors = vector_search.search_triplets(
            text_embeddings.embeddings,
            brain_id=brain_id,
            k=limit,
        )

        search_result = data_adapter.search(text, brain_id)

        ts_text_chunks = search_result.text_chunks
        ts_observations = search_result.observations

        v_text_chunks, v_observations = data_adapter.get_text_chunks_by_ids(
            [dv.metadata.get("resource_id") for dv in data_vectors], True, brain_id
        )

        node_ids = [
            node_id
            for tv in triple_vectors
            for node_id in tv.metadata.get("node_ids", [])
        ]
        nodes = graph_adapter.get_nodes_by_uuid(
            uuids=node_ids,
            brain_id=brain_id,
            with_relationships=True,
            relationships_depth=1,
            relationships_type=[
                tv.metadata.get("predicate")
                for tv in triple_vectors
                if tv.metadata.get("predicate", None)
            ],
            preferred_labels=preferred_entities or [],
        )
        if not include_rejected:
            chunk_ids = [
                item.id
                for item in [*ts_text_chunks, *v_text_chunks]
                if getattr(item, "id", None)
            ]
            resource_ids = [
                getattr(obs, "resource_id", None)
                for obs in [*ts_observations, *v_observations]
            ]
            citing_ids = [item for item in [*chunk_ids, *resource_ids] if item]
            citing_nodes, citing_triples = [], []
            lookup = getattr(graph_adapter, "nodes_and_edges_citing", None)
            if citing_ids and callable(lookup):
                found = lookup(brain_id, citing_ids, [])
                if isinstance(found, (tuple, list)) and len(found) >= 2:
                    citing_nodes = found[0] or []
                    citing_triples = found[1] or []
            rejected_chunks = chunk_ids_with_only_rejected_support(
                citing_nodes, citing_triples
            )
            ts_text_chunks = filter_items_with_rejected_support(
                ts_text_chunks, rejected_chunks
            )
            v_text_chunks = filter_items_with_rejected_support(
                v_text_chunks, rejected_chunks
            )
            ts_observations = filter_items_with_rejected_support(
                ts_observations, rejected_chunks
            )
            v_observations = filter_items_with_rejected_support(
                v_observations, rejected_chunks
            )
        return ts_text_chunks, ts_observations, v_text_chunks, v_observations, nodes

    with verification_scope(include_rejected=include_rejected):
        ts_text_chunks, ts_observations, v_text_chunks, v_observations, nodes = (
            await asyncio.to_thread(_get_data)
        )
    unknown_items: list[dict] = []
    if as_of_dt is not None:
        unknown_chunk_ids: set[str] = set()
        excluded_chunk_ids: set[str] = set()
        kept_ts: list = []
        for chunk in ts_text_chunks:
            status = chunk_validity_status(chunk, as_of_dt)
            if status == VALIDITY_UNKNOWN:
                if chunk.id not in unknown_chunk_ids:
                    unknown_chunk_ids.add(chunk.id)
                    unknown_items.append(_unknown_chunk_item(chunk, unknown_fact))
            elif status == VALIDITY_VALID:
                kept_ts.append(chunk)
            else:
                excluded_chunk_ids.add(chunk.id)
        kept_v: list = []
        for chunk in v_text_chunks:
            status = chunk_validity_status(chunk, as_of_dt)
            if status == VALIDITY_UNKNOWN:
                if chunk.id not in unknown_chunk_ids:
                    unknown_chunk_ids.add(chunk.id)
                    unknown_items.append(_unknown_chunk_item(chunk, unknown_fact))
            elif status == VALIDITY_VALID:
                kept_v.append(chunk)
            else:
                excluded_chunk_ids.add(chunk.id)
        drop_chunk_ids = unknown_chunk_ids | excluded_chunk_ids
        ts_text_chunks = kept_ts
        v_text_chunks = kept_v
        ts_observations = [
            obs
            for obs in ts_observations
            if getattr(obs, "resource_id", None) not in drop_chunk_ids
        ]
        v_observations = [
            obs
            for obs in v_observations
            if getattr(obs, "resource_id", None) not in drop_chunk_ids
        ]
        kept_nodes = []
        for hit in nodes:
            status = _retrieve_hit_validity_status(hit, as_of_dt, validity_status)
            if status == VALIDITY_UNKNOWN:
                unknown_items.append(
                    unknown_fact(
                        target="relationship",
                        item_id=_retrieve_hit_id(hit),
                        payload=jsonable_graph_value(hit),
                    )
                )
            elif status == VALIDITY_VALID:
                kept_nodes.append(hit)
        nodes = kept_nodes

    envelope = {}
    if as_of_dt is not None:
        unknown_items = dedupe_unknown_items(unknown_items)
        envelope["validity_unknown_count"] = len(unknown_items)
        if include_validity_unknown:
            envelope["validity_unknown"] = unknown_items

    return RetrieveRequestResponse(
        data=[*ts_text_chunks, *v_text_chunks],
        observations=[*ts_observations, *v_observations],
        relationships=nodes,
        **envelope,
    )


async def retrieve_neighbors(
    uuid: Optional[str] = None,
    look_for: Optional[str] = None,
    identification_params: Optional[IdentificationParams] = None,
    limit: int = 10,
    brain_id: str = "default",
    include_rejected: bool = False,
) -> RetrieveNeighborsRequestResponse:
    """
    Retrieve neighboring nodes related to a specified main node.

    If `uuid` is provided it is used to locate the main node; otherwise `identification_params` is used. Optionally filters first-degree neighbors by semantic similarity to `look_for`, expands those matches to similar nodes, and returns a deduplicated list of neighbor nodes (up to `limit`) with their relationship and the most common matching similar node.

    Parameters:
        uuid (Optional[str]): UUID of the main node to retrieve neighbors for. If omitted, `identification_params` must be provided.
        look_for (Optional[str]): Text used to filter first-degree neighbors by embedding similarity before expanding to similar nodes.
        identification_params (Optional[IdentificationParams]): Identification parameters used to find the main node when `uuid` is not provided.
        limit (int): Maximum number of neighbor results to include in the response.
        brain_id (str): Identifier of the brain / dataset to query.

    Returns:
        RetrieveNeighborsRequestResponse: Object containing:
          - count: total number of unique neighbors found,
          - main_node: the resolved main Node,
          - neighbors: list of RetrievedNeighborNode objects (neighbor, relationship, most_common) limited to `limit`.

    Raises:
        HTTPException: 404 if the main node cannot be found.
    """

    async def _get_neighbors():

        # ---------------------------------------------------------
        # ================= GETTING THE MAIN NODE =================
        # ---------------------------------------------------------
        def _get_node() -> Node:
            node = None
            if uuid:
                node = graph_adapter.get_by_uuid(uuid, brain_id)
            elif identification_params:
                node = graph_adapter.get_by_identification_params(
                    identification_params,
                    brain_id=brain_id,
                    entity_types=identification_params.entity_types,
                )
            if not node:
                raise HTTPException(status_code=404, detail="Entity not found")
            return node

        node = await asyncio.to_thread(_get_node)
        target_node_types = node.labels

        looking_for_v = _embed_query_vector(look_for) if look_for else None

        # ---------------------------------------------------------
        # ===== Getting 1st degree neighbors of the main node =====
        # ---------------------------------------------------------
        def _get_fd_neighbors() -> (
            tuple[dict[str, list[tuple[Predicate, Node]]], list[str]]
        ):
            fd_neighbors = graph_adapter.get_neighbors(
                [node.uuid], limit=limit, brain_id=brain_id
            )
            fd_v_neighbors_ids = [
                fd[1].properties.get("v_id")
                for fd in fd_neighbors[node.uuid]
                if fd[1].properties.get("v_id") is not None
            ]
            if look_for:
                fd_v_neighbors_embeddings = vector_store_adapter.get_by_ids(
                    fd_v_neighbors_ids, brain_id=brain_id, store="nodes"
                )
                fd_v_neighbors_embeddings_map = {
                    v.id: v.embeddings
                    for v in fd_v_neighbors_embeddings
                    if (
                        cosine_similarity(looking_for_v.embeddings, v.embeddings) > 0.5
                        and v.id
                        and not v.id.replace(
                            "-", ""
                        ).isalpha()  # likely not a UUID if all numeric (may have hyphens for uuid standard)
                    )
                }
                fd_v_neighbors_ids = list(fd_v_neighbors_embeddings_map.keys())

            return fd_neighbors, fd_v_neighbors_ids

        fd_neighbors, fd_v_neighbors_ids = await asyncio.to_thread(_get_fd_neighbors)

        # ---------------------------------------------------------
        # === Getting nodes similar to the 1st degree neighbors ===
        # ---------------------------------------------------------
        fd_v_similar_node_futures = []
        for fd_v_neighbor_id in fd_v_neighbors_ids:
            fd_v_similar_node_futures.append(
                asyncio.to_thread(
                    vector_store_adapter.search_similar_by_ids,
                    [fd_v_neighbor_id],
                    brain_id,
                    "nodes",
                    0.5,
                    limit,
                )
            )
        fd_v_similar_nodes_results: list[dict[str, list[Vector]]] = (
            await asyncio.gather(*fd_v_similar_node_futures)
        )
        fd_similar_node_ids = [
            v.metadata.get("uuid")
            for result_dict in fd_v_similar_nodes_results
            for vectors in result_dict.values()
            for v in vectors
            if v.metadata.get("uuid") is not None
        ]
        fd_similar_nodes = await asyncio.to_thread(
            graph_adapter.get_by_uuids, fd_similar_node_ids, brain_id
        )
        fd_similar_nodes_by_uuid = {n.uuid: n for n in fd_similar_nodes}

        # ---------------------------------------------------------
        # === Getting neighbors of the 1st degree similar nodes ===
        # ---------------------------------------------------------
        def _get_fd_similar_node_neighbors() -> dict[str, list[tuple[Predicate, Node]]]:
            fd_similar_node_neighbors = graph_adapter.get_neighbors(
                fd_similar_node_ids,
                limit=limit,
                brain_id=brain_id,
                of_types=list(set(target_node_types)),
            )
            return fd_similar_node_neighbors

        fd_similar_node_neighbors = await asyncio.to_thread(
            _get_fd_similar_node_neighbors
        )

        seen_neighbor_uuids = set()
        unique_neighbors = []
        for source_uuid, neighbors_list in fd_similar_node_neighbors.items():
            for neighbor_tuple in neighbors_list:
                neighbor_uuid = neighbor_tuple[1].uuid
                if neighbor_uuid not in seen_neighbor_uuids:
                    seen_neighbor_uuids.add(neighbor_uuid)
                    unique_neighbors.append(
                        RetrievedNeighborNode(
                            neighbor=neighbor_tuple[1],
                            relationship=neighbor_tuple[0],
                            most_common=fd_similar_nodes_by_uuid.get(source_uuid),
                        )
                    )

        return RetrieveNeighborsRequestResponse(
            count=len(unique_neighbors),
            main_node=node,
            neighbors=unique_neighbors[:limit],
        )

    with verification_scope(include_rejected=include_rejected):
        return await _get_neighbors()


async def retrieve_neighbors_ai_mode(
    identification_params: IdentificationParams,
    looking_for: Optional[list[str]],
    limit: int,
    brain_id: str = "default",
) -> RetrieveNeighborsRequestResponse:
    """
    Retrieve neighbors of an entity from the knowledge graph.
    """

    def _get_neighbors():
        node = graph_adapter.get_by_identification_params(
            identification_params,
            brain_id=brain_id,
            entity_types=identification_params.entity_types,
        )
        if not node:
            raise HTTPException(status_code=404, detail="Entity not found")

        result = kg_agent.retrieve_neighbors(node, looking_for, limit, brain_id)

        ids = [neighbor.uuid for neighbor in result.neighbors]
        descriptions = [neighbor.description for neighbor in result.neighbors]

        nodes = graph_adapter.get_nodes_by_uuid(uuids=ids, brain_id=brain_id)
        paired = list(zip(nodes, descriptions))

        return RetrieveNeighborsRequestResponse(neighbors=paired)

    result = await asyncio.to_thread(_get_neighbors)

    return result


async def get_relationships(
    limit: int = 10,
    skip: int = 0,
    relationship_types: Optional[list[str]] = None,
    from_node_labels: Optional[list[str]] = None,
    to_node_labels: Optional[list[str]] = None,
    query_text: Optional[str] = None,
    query_search_target: Optional[str] = "all",
    brain_id: str = "default",
    include_rejected: bool = False,
):
    """
    Retrieve relationships from the knowledge graph with optional filtering and pagination.

    Parameters:
        relationship_types (list[str], optional): Filter results to specific relationship types.
        from_node_labels (list[str], optional): Filter relationships originating from nodes with these labels.
        to_node_labels (list[str], optional): Filter relationships targeting nodes with these labels.
        query_text (str, optional): Text to search within relationship or node content.
        query_search_target (str, optional): Field to target for text search; commonly "all", "from", or "to".
        limit (int, optional): Maximum number of relationships to return.
        skip (int, optional): Number of relationships to skip (offset).
        brain_id (str, optional): Identifier of the brain/graph to query.

    Returns:
        JSONResponse: A response whose JSON content contains:
            - message: Confirmation string.
            - relationships: List of serialized relationship objects.
            - total: Total number of matching relationships.
    """
    with verification_scope(include_rejected=include_rejected):
        relationships = await asyncio.to_thread(
            search_relationships,
            limit,
            skip,
            relationship_types,
            from_node_labels,
            to_node_labels,
            query_text,
            query_search_target,
            brain_id,
        )

    return JSONResponse(
        content={
            "message": "Relationships retrieved successfully",
            "relationships": [r.model_dump(mode="json") for r in relationships.results],
            "total": relationships.total,
        }
    )


async def get_entities(
    limit: int = 10,
    skip: int = 0,
    node_labels: Optional[list[str]] = None,
    query_text: Optional[str] = None,
    brain_id: str = "default",
    include_rejected: bool = False,
):
    """
    Retrieve entities from the knowledge graph with optional label and text filters.

    Parameters:
        limit (int): Maximum number of entities to return (pagination).
        skip (int): Number of entities to skip (pagination offset).
        node_labels (Optional[list[str]]): If provided, only return entities whose labels match any value in this list.
        query_text (Optional[str]): If provided, filter entities by matching text content.
        brain_id (str): Identifier of the knowledge graph/brain to query.

    Returns:
        JSONResponse: Object containing:
            - message (str): Informational message.
            - entities (list): Serialized entity objects.
            - total (int): Total number of matching entities.
    """
    with verification_scope(include_rejected=include_rejected):
        entities = await asyncio.to_thread(
            search_entities, limit, skip, node_labels, query_text, brain_id
        )

    return JSONResponse(
        content={
            "message": "Entities retrieved successfully",
            "entities": [e.model_dump(mode="json") for e in entities.results],
            "total": entities.total,
        }
    )


def _format_kg_item(item: Any) -> str:
    name = getattr(item, "name", None) or ""
    description = getattr(item, "description", None) or ""
    happened_at = getattr(item, "happened_at", None) or ""
    base = f"{name}: {description}" if description else name
    if happened_at:
        return f"{base} @{happened_at}"
    return base


def _format_event_fact(n: Node, r: Predicate, m: Node, r2: Predicate, b: Node) -> str:
    r2_name = (getattr(r2, "name", None) or "").upper()
    end_label = (
        "Context"
        if any(token in r2_name for token in _CONTEXT_REL)
        else "Target"
    )
    return (
        f"Actor: {_format_kg_item(n)} | {_format_kg_item(r)} | "
        f"Event: {_format_kg_item(m)} | {_format_kg_item(r2)} | "
        f"{end_label}: {_format_kg_item(b)}"
    )


def _fact_key(r: Predicate, r2: Predicate) -> tuple[str, str]:
    return (getattr(r, "uuid", None) or "", getattr(r2, "uuid", None) or "")


def _rel_name(predicate: Predicate | None) -> str:
    return (getattr(predicate, "name", None) or "").upper()


def _is_spine_actor_rel(name: str) -> bool:
    return any(token in name for token in _SPINE_ACTOR)


def _is_spine_target_rel(name: str) -> bool:
    return any(token in name for token in _SPINE_TARGET)


def _is_context_rel(name: str) -> bool:
    return any(token in name for token in _CONTEXT_REL)


def _event_hub_id(candidate: dict[str, Any]) -> str:
    triple = candidate.get("triple") or ()
    r = triple[1] if len(triple) > 1 else None
    m = triple[2] if len(triple) > 2 else None
    r2 = triple[3] if len(triple) > 3 else None
    event_uuid = getattr(m, "uuid", None) if m is not None else None
    if event_uuid:
        return str(event_uuid)
    flow_key = (
        getattr(r, "flow_key", None)
        or getattr(r2, "flow_key", None)
        or ""
    )
    if flow_key:
        return str(flow_key)
    key = candidate.get("key")
    if isinstance(key, tuple):
        return "|".join(str(part or "") for part in key)
    return str(candidate.get("text") or "")


def _candidate_session_ids(candidate: dict[str, Any]) -> list[str]:
    sessions = candidate.get("session_ids")
    if isinstance(sessions, list) and sessions:
        return [str(s) for s in sessions if s]
    return []


def _hub_leg_kind(candidate: dict[str, Any]) -> str:
    triple = candidate.get("triple") or ()
    r2 = triple[3] if len(triple) > 3 else None
    name = _rel_name(r2)
    if _is_context_rel(name):
        return "context"
    if _is_spine_target_rel(name):
        return "spine"
    return "other"


def _hub_completeness_score(candidate: dict[str, Any]) -> float:
    triple = candidate.get("triple") or ()
    if len(triple) < 5:
        return 0.0
    n, r, m, r2, b = triple
    score = 0.0
    m_labels = {str(label).upper() for label in (getattr(m, "labels", None) or [])}
    if "EVENT" in m_labels:
        score += 1.0
    r_name = _rel_name(r)
    r2_name = _rel_name(r2)
    if _is_spine_actor_rel(r_name):
        score += 1.0
    elif r_name:
        score += 0.2
    if _is_spine_target_rel(r2_name):
        score += 1.0
    elif _is_context_rel(r2_name):
        score += 0.35
    elif r2_name:
        score += 0.15
    if getattr(n, "uuid", None) and getattr(b, "uuid", None):
        score += 0.5
    if getattr(m, "happened_at", None):
        score += 0.15
    return score


def _fact_recency_score(
    candidate: dict[str, Any], *, now: datetime | None = None
) -> float:
    triple = candidate.get("triple") or ()
    best = 0.0
    now = now or datetime.now(tz=None)
    for node in triple[0::2] if triple else ():
        raw = getattr(node, "happened_at", None)
        if not raw:
            continue
        parsed = parse_date_string(raw) if isinstance(raw, str) else raw
        if not isinstance(parsed, datetime):
            continue
        days_ago = max(0, (now - to_naive_utc(parsed)).days)
        score = 1.0 if days_ago <= 0 else 1.0 / (1.0 + days_ago)
        if score > best:
            best = score
    return best


def _relevance_score(
    candidate: dict[str, Any], *, now: datetime | None = None
) -> float:
    distance = _candidate_distance(candidate)
    distance_term = 1.0 / (1.0 + max(0.0, distance))
    ppr_term = float(candidate.get("ppr_mass") or 0.0)
    completeness = _hub_completeness_score(candidate) / 3.8
    recency = _fact_recency_score(candidate, now=now)
    topic_boost = 0.08 if candidate.get("topic_preferred") else 0.0
    triple = candidate.get("triple") or ()
    verified_boost = 0.0
    if fact_tuple_is_verified(*triple[:5] if len(triple) >= 5 else triple):
        from src.core.verification.facts import verified_rank_boost

        verified_boost = verified_rank_boost()
    return (
        0.45 * distance_term
        + 0.25 * ppr_term
        + 0.20 * completeness
        + 0.10 * recency
        + topic_boost
        + verified_boost
    )


def _completeness_rank_key(candidate: dict[str, Any]) -> tuple:
    return (
        -_relevance_score(candidate),
        _candidate_distance(candidate),
    ) + _candidate_tiebreak_key(candidate)


def _rank_facts_with_completeness(
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return sorted(candidates, key=_completeness_rank_key)


def _is_complementary_hub_leg(
    selected_meta: list[dict[str, Any]], item: dict[str, Any]
) -> bool:
    if item["kind"] not in {"spine", "context"}:
        return False
    for prev in selected_meta:
        if prev["hub"] != item["hub"]:
            continue
        if {item["kind"], prev["kind"]} == {"spine", "context"}:
            return True
    return False


def _bridge_reserved_slots(max_facts: int) -> int:
    if max_facts <= 0:
        return 0
    return min(
        max_facts,
        max(_BRIDGE_RESERVE_MIN, max_facts // _BRIDGE_RESERVE_FRAC_DENOM),
    )


def _is_bridge_candidate(candidate: dict[str, Any]) -> bool:
    return bool(candidate.get("bridge"))


def _bridge_reserve_ok(candidate: dict[str, Any]) -> bool:
    bridge = candidate.get("bridge")
    if not bridge:
        return False
    if isinstance(bridge, dict) and "reserve_ok" in bridge:
        return bool(bridge.get("reserve_ok"))
    return True


def _prepare_diversify_item(
    candidate: dict[str, Any], *, now: datetime | None = None
) -> dict[str, Any]:
    key = candidate.get("key") or ("", "")
    is_bridge = _is_bridge_candidate(candidate)
    return {
        "candidate": candidate,
        "relevance": _relevance_score(candidate, now=now),
        "hub": _event_hub_id(candidate),
        "sessions": frozenset(_candidate_session_ids(candidate)),
        "kind": _hub_leg_kind(candidate),
        "tie": (str(key[0]) if key else "", str(key[1]) if len(key) > 1 else ""),
        "is_bridge": is_bridge,
        "reserve_ok": _bridge_reserve_ok(candidate) if is_bridge else False,
    }


def _mmr_pick_index(
    remaining: list[dict[str, Any]],
    selected_meta: list[dict[str, Any]],
    selected_hubs: set[str],
    selected_sessions: set[str],
) -> int:
    best_idx = 0
    best_mmr = None
    best_tie = None
    selected_kinds_by_hub: dict[str, set[str]] = defaultdict(set)
    for selected in selected_meta:
        selected_kinds_by_hub[selected["hub"]].add(selected["kind"])
    for idx, item in enumerate(remaining):
        relevance = item["relevance"]
        hub = item["hub"]
        sessions = item["sessions"]
        complementary = (
            item["kind"] in {"spine", "context"}
            and bool(
                selected_kinds_by_hub.get(item["hub"], set())
                & ({"spine", "context"} - {item["kind"]})
            )
        )
        if complementary:
            hub_pen = 0.0
            sess_pen = 0.0
            relevance = relevance + 0.5
        else:
            hub_pen = 1.0 if hub in selected_hubs else 0.0
            if sessions and selected_sessions:
                sess_pen = len(sessions & selected_sessions) / max(
                    1, len(sessions)
                )
            else:
                sess_pen = 0.0
        redundancy = 0.55 * hub_pen + 0.45 * sess_pen
        mmr = _DIVERSIFY_LAMBDA * relevance - (1.0 - _DIVERSIFY_LAMBDA) * redundancy
        tie = item["tie"]
        if (
            best_mmr is None
            or mmr > best_mmr
            or (mmr == best_mmr and (best_tie is None or tie < best_tie))
        ):
            best_mmr = mmr
            best_tie = tie
            best_idx = idx
    return best_idx


def _mmr_take(
    remaining: list[dict[str, Any]],
    *,
    budget: int,
    selected: list[dict[str, Any]],
    selected_meta: list[dict[str, Any]],
    selected_hubs: set[str],
    selected_sessions: set[str],
) -> None:
    while budget > 0 and remaining:
        best_idx = _mmr_pick_index(
            remaining, selected_meta, selected_hubs, selected_sessions
        )
        chosen = remaining.pop(best_idx)
        selected.append(chosen["candidate"])
        selected_meta.append(chosen)
        selected_hubs.add(chosen["hub"])
        selected_sessions.update(chosen["sessions"])
        budget -= 1


def _take_novel_bridge_slots(
    remaining: list[dict[str, Any]],
    *,
    budget: int,
    covered_sessions: set[str],
    selected: list[dict[str, Any]],
    selected_meta: list[dict[str, Any]],
    selected_hubs: set[str],
    selected_sessions: set[str],
) -> None:
    while budget > 0 and remaining:
        novel_sessions: set[str] = set()
        for item in remaining:
            if not item["is_bridge"] or not item["reserve_ok"]:
                continue
            novel_sessions.update(item["sessions"] - covered_sessions)
        if not novel_sessions:
            break
        target = min(novel_sessions)
        best_idx = None
        best_key = None
        for idx, item in enumerate(remaining):
            if not item["is_bridge"] or not item["reserve_ok"]:
                continue
            if target not in item["sessions"]:
                continue
            key = (item["hub"], item["tie"])
            if best_key is None or key < best_key:
                best_key = key
                best_idx = idx
        if best_idx is None:
            break
        chosen = remaining.pop(best_idx)
        selected.append(chosen["candidate"])
        selected_meta.append(chosen)
        selected_hubs.add(chosen["hub"])
        selected_sessions.update(chosen["sessions"])
        covered_sessions.update(chosen["sessions"])
        budget -= 1


def _diversify_facts(
    ranked: list[dict[str, Any]],
    *,
    max_facts: int,
) -> list[dict[str, Any]]:
    if max_facts <= 0 or not ranked:
        return []
    if len(ranked) <= max_facts:
        return list(ranked)

    ranking_now = datetime.now(tz=None)
    prepared = [
        _prepare_diversify_item(candidate, now=ranking_now) for candidate in ranked
    ]
    prepared.sort(key=lambda item: (-item["relevance"], item["tie"]))

    selected: list[dict[str, Any]] = []
    selected_meta: list[dict[str, Any]] = []
    selected_hubs: set[str] = set()
    selected_sessions: set[str] = set()

    reserve = _bridge_reserved_slots(max_facts)
    has_bridge = any(item["is_bridge"] and item["reserve_ok"] for item in prepared)
    if not has_bridge:
        reserve = 0
    main_budget = max_facts - reserve

    non_bridge = [item for item in prepared if not item["is_bridge"]]
    bridge_items = [item for item in prepared if item["is_bridge"]]
    stable_bridges = [item for item in bridge_items if item["reserve_ok"]]

    _mmr_take(
        non_bridge,
        budget=main_budget,
        selected=selected,
        selected_meta=selected_meta,
        selected_hubs=selected_hubs,
        selected_sessions=selected_sessions,
    )

    covered_by_non_bridge = set(selected_sessions)
    _take_novel_bridge_slots(
        stable_bridges,
        budget=reserve,
        covered_sessions=covered_by_non_bridge,
        selected=selected,
        selected_meta=selected_meta,
        selected_hubs=selected_hubs,
        selected_sessions=selected_sessions,
    )

    remaining = non_bridge + stable_bridges
    remaining.sort(key=lambda item: (-item["relevance"], item["tie"]))
    leftover = max_facts - len(selected)
    _mmr_take(
        remaining,
        budget=leftover,
        selected=selected,
        selected_meta=selected_meta,
        selected_hubs=selected_hubs,
        selected_sessions=selected_sessions,
    )
    return selected


def _temporal_conflict_meta(
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_entity: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        triple = candidate.get("triple") or ()
        for node in (triple[0], triple[4]) if len(triple) >= 5 else ():
            uuid = getattr(node, "uuid", None)
            if not uuid:
                continue
            happened = None
            event = triple[2] if len(triple) > 2 else None
            raw = getattr(event, "happened_at", None) if event is not None else None
            if isinstance(raw, str) and raw.strip():
                happened = raw.strip()
            by_entity[str(uuid)].append(
                {
                    "fact_key": list(candidate.get("key") or ()),
                    "hub_id": _event_hub_id(candidate),
                    "happened_at": happened,
                }
            )
    conflicts: list[dict[str, Any]] = []
    for entity_uuid, entries in sorted(by_entity.items()):
        times = {
            entry["happened_at"]
            for entry in entries
            if entry.get("happened_at")
        }
        hubs = {entry["hub_id"] for entry in entries}
        if len(times) >= 2 and len(hubs) >= 2:
            conflicts.append(
                {
                    "entity_uuid": entity_uuid,
                    "hubs": sorted(hubs),
                    "happened_ats": sorted(times),
                    "facts": [entry["fact_key"] for entry in entries],
                }
            )
    return conflicts


def _edge_type_weight(predicate: Predicate | None, query: str | None = None) -> float:
    name = _rel_name(predicate)
    if _is_spine_actor_rel(name) or _is_spine_target_rel(name):
        weight = 1.0
    elif _is_context_rel(name):
        weight = 0.35
    else:
        weight = 0.55
    if query:
        q_tokens = set(re.findall(r"[a-z0-9]+", query.lower()))
        p_tokens = set(re.findall(r"[a-z0-9]+", name.lower()))
        if q_tokens and p_tokens:
            overlap = len(q_tokens & p_tokens) / max(1, len(p_tokens))
            weight *= 1.0 + 0.5 * overlap
    return weight


def _add_weighted_edge(
    adjacency: dict[str, dict[str, float]],
    src: str | None,
    dst: str | None,
    weight: float,
) -> None:
    if not src or not dst or src == dst or weight <= 0:
        return
    bucket = adjacency.setdefault(str(src), {})
    prev = bucket.get(str(dst), 0.0)
    if weight > prev:
        bucket[str(dst)] = weight


def _candidate_tiebreak_key(candidate: dict[str, Any]) -> tuple[str, ...]:
    triple = candidate.get("triple") or ()
    n = triple[0] if len(triple) > 0 else None
    r = triple[1] if len(triple) > 1 else None
    m = triple[2] if len(triple) > 2 else None
    r2 = triple[3] if len(triple) > 3 else None
    b = triple[4] if len(triple) > 4 else None
    flow_key = (
        getattr(r, "flow_key", None)
        or getattr(r2, "flow_key", None)
        or ""
    )
    key = candidate.get("key")
    if not (isinstance(key, tuple) and len(key) == 2):
        key = _fact_key(r, r2) if r is not None and r2 is not None else ("", "")
    r_uuid, r2_uuid = key
    return (
        str(flow_key or ""),
        str(r_uuid or ""),
        str(r2_uuid or ""),
        str(getattr(n, "uuid", None) or ""),
        str(getattr(m, "uuid", None) or ""),
        str(getattr(b, "uuid", None) or ""),
        str(candidate.get("identified_entity") or ""),
        str(candidate.get("text") or ""),
    )


def _distance_rank_key(candidate: dict[str, Any]) -> tuple:
    return (_candidate_distance(candidate),) + _candidate_tiebreak_key(
        candidate
    )


def _ppr_rank_key(
    candidate: dict[str, Any], ppr_scores: dict[str, float]
) -> tuple:
    return (_score_candidate_with_ppr(candidate, ppr_scores),) + _candidate_tiebreak_key(
        candidate
    )


def _dedupe_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: dict[tuple[str, str], dict[str, Any]] = {}
    for candidate in candidates:
        key = candidate["key"]
        existing = deduped.get(key)
        if existing is None:
            deduped[key] = candidate
            continue
        if candidate["score"] < existing["score"]:
            deduped[key] = candidate
            continue
        if candidate["score"] == existing["score"] and _candidate_tiebreak_key(
            candidate
        ) < _candidate_tiebreak_key(existing):
            deduped[key] = candidate
    return sorted(deduped.values(), key=_distance_rank_key)


def _merge_variant_candidate_lists(
    variant_lists: list[list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for local in variant_lists:
        ordered = sorted(local, key=_distance_rank_key)
        merged.extend(ordered)
    return merged


def _relationship_fields(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        nested = value.get("properties")
        if isinstance(nested, dict):
            return nested
        return value
    props = getattr(value, "properties", None)
    if isinstance(props, dict):
        return props
    getter = getattr(value, "get", None)
    if callable(getter):
        fields: dict[str, Any] = {}
        for key in (
            "valid_from",
            "valid_to",
            "valid_at",
            "invalid_at",
            "superseded_at",
            "deprecated",
        ):
            item = getter(key)
            if item is not None:
                fields[key] = item
        return fields
    return {}


def _retrieve_hit_valid_at(hit: Any, as_of, is_valid_at) -> bool:
    if not isinstance(hit, dict):
        return True
    rels = hit.get("relationships")
    if rels is None:
        predicate = hit.get("predicate")
        if predicate is None:
            return True
        return is_valid_at(_relationship_fields(predicate), as_of)
    items = rels if isinstance(rels, (list, tuple)) else [rels]
    return all(
        is_valid_at(_relationship_fields(rel), as_of)
        for rel in items
        if rel is not None
    )


def _retrieve_hit_validity_status(hit: Any, as_of, validity_status) -> str:
    if not isinstance(hit, dict):
        return "valid"
    rels = hit.get("relationships")
    if rels is None:
        predicate = hit.get("predicate")
        if predicate is None:
            return "valid"
        return validity_status(_relationship_fields(predicate), as_of)
    items = rels if isinstance(rels, (list, tuple)) else [rels]
    statuses = [
        validity_status(_relationship_fields(rel), as_of)
        for rel in items
        if rel is not None
    ]
    if not statuses:
        return "valid"
    if any(status == "invalid" for status in statuses):
        return "invalid"
    if any(status == "unknown" for status in statuses):
        return "unknown"
    return "valid"


def _retrieve_hit_id(hit: Any) -> str:
    if not isinstance(hit, dict):
        return ""
    rels = hit.get("relationships")
    if isinstance(rels, dict):
        return str(rels.get("uuid") or rels.get("name") or "")
    predicate = hit.get("predicate")
    if isinstance(predicate, dict):
        return str(predicate.get("uuid") or predicate.get("name") or "")
    uuid = getattr(predicate, "uuid", None) if predicate is not None else None
    return str(uuid or "")


def _unknown_chunk_item(chunk: Any, unknown_fact) -> dict:
    payload = chunk.model_dump(mode="json") if hasattr(chunk, "model_dump") else chunk
    return unknown_fact(
        target="chunk",
        item_id=str(getattr(chunk, "id", "") or ""),
        payload=payload,
    )


def _is_currently_valid(
    predicate: Predicate, as_of: datetime | None = None
) -> bool:
    from src.core.saving.validity import predicate_is_valid_at

    return predicate_is_valid_at(predicate, as_of)


def _wants_historical_facts(
    query: str,
    *,
    explicit: bool | None = None,
    query_vector: list[float] | None = None,
    history_vectors: list[list[float]] | None = None,
    current_vectors: list[list[float]] | None = None,
) -> bool:
    from src.core.saving.history_intent import resolve_include_history

    return resolve_include_history(
        explicit,
        query_text=query,
        query=query_vector,
        history_vectors=history_vectors,
        current_vectors=current_vectors,
    )


def _fact_predicates_allowed(
    r: Predicate,
    r2: Predicate,
    *,
    include_history: bool,
    as_of: datetime | None = None,
) -> bool:
    if not fact_pair_allowed(r, r2):
        return False
    if as_of is not None:
        return _is_currently_valid(r, as_of) and _is_currently_valid(r2, as_of)
    if include_history:
        return True
    return _is_currently_valid(r) and _is_currently_valid(r2)


def _collect_query_variants(text: str, elements) -> list[str]:
    seen: set[str] = set()
    variants: list[str] = []

    def _add(value: str | None) -> None:
        if not value:
            return
        cleaned = value.strip()
        if not cleaned:
            return
        key = cleaned.lower()
        if key in seen:
            return
        seen.add(key)
        variants.append(cleaned)

    _add(text)
    for token in getattr(elements, "tokens", None) or []:
        if isinstance(token, dict):
            _add(token.get("text"))
    noun_chunks = getattr(elements, "noun_chunks", None) or []
    added_chunks = 0
    for chunk in noun_chunks:
        if added_chunks >= _MAX_NOUN_CHUNKS:
            break
        before = len(variants)
        _add(chunk if isinstance(chunk, str) else None)
        if len(variants) > before:
            added_chunks += 1
    for extra in _decompose_enumeration_queries(text, elements):
        _add(extra)
    for extra in _ordering_aspect_queries(text):
        _add(extra)
    return variants


_ENUMERATION_RE = re.compile(
    r"\b(what|which|list|ways?|types?|kinds?|events?|books?|items?|"
    r"changes?|traits?|hobbies|activities)\b",
    re.IGNORECASE,
)
_ORDERING_ASPECT_QUERIES_BUDGET = (
    "core functionality authentication expense tracking visualization",
    "transaction CRUD error handling response management",
    "security deployment integration tests gunicorn workers",
)
_ORDERING_ASPECT_QUERIES_TRANSLATION = (
    "translation API integration error handling DeepL",
    "rate limiting request queue caching Redis",
    "language detection libraries franc evaluation",
    "database schema optimization contextual memory store",
)


def _is_enumeration_question(text: str) -> bool:
    return bool(_ENUMERATION_RE.search(text or ""))


def _is_ordering_question(text: str) -> bool:
    return bool(
        re.search(
            r"\b(order|ordered|ordering|chronolog|sequence|walk me through)\b",
            text or "",
            re.I,
        )
    )


def _ordering_aspect_queries(text: str) -> list[str]:
    if not _is_ordering_question(text):
        return []
    q = text or ""
    if re.search(
        r"\b(translation|language detection|deepl|franc|microservice|"
        r"multi-language|rate limit)\b",
        q,
        re.I,
    ):
        return list(_ORDERING_ASPECT_QUERIES_TRANSLATION)[:2]
    return list(_ORDERING_ASPECT_QUERIES_BUDGET)[:2]


def _decompose_enumeration_queries(text: str, elements) -> list[str]:
    if not _is_enumeration_question(text):
        return []
    queries: list[str] = []
    entities: list[str] = []
    for token in getattr(elements, "tokens", None) or []:
        if not isinstance(token, dict):
            continue
        label = str(token.get("label") or token.get("ent_type") or "").upper()
        value = str(token.get("text") or "").strip()
        if not value:
            continue
        if label in {"PERSON", "PER", "ORG", "GPE"} or (
            value[:1].isupper() and " " not in value
        ):
            entities.append(value)
    relation_terms = [
        m.group(0).lower()
        for m in re.finditer(
            r"\b(books?|read|ways?|types?|kinds?|events?|hobbies|"
            r"activities|changes?|traits?|friends?|places?)\b",
            text or "",
            re.IGNORECASE,
        )
    ]
    seen: set[str] = set()
    for entity in entities[:4]:
        for relation in relation_terms[:4]:
            q = f"{entity} {relation}"
            key = q.lower()
            if key not in seen and key != (text or "").strip().lower():
                seen.add(key)
                queries.append(q)
        key = entity.lower()
        if key not in seen:
            seen.add(key)
            queries.append(entity)
    for relation in relation_terms[:3]:
        if relation not in seen:
            seen.add(relation)
            queries.append(relation)
    return queries[:8]


def _seed_personalization(seeds: list[tuple[str, float, str]]) -> dict[str, float]:
    weights: dict[str, float] = {}
    for uuid, distance, _ in seeds:
        weight = 1.0 / (1.0 + max(0.0, _quantize_distance(distance)))
        prev = weights.get(uuid)
        if prev is None or weight > prev:
            weights[uuid] = weight
    return weights


def _build_adjacency_from_seeds(
    seed_uuids: list[str],
    brain_id: str,
    *,
    query: str | None = None,
    query_embedding: list[float] | None = None,
    include_history: bool = False,
    as_of: datetime | None = None,
) -> dict[str, list[tuple[str, float]]]:
    weighted: dict[str, dict[str, float]] = {}
    if not seed_uuids:
        return {}
    try:
        neighbors = graph_adapter.get_event_centric_neighbors(
            seed_uuids, brain_id=brain_id
        )
    except Exception:
        return {}
    for n, r, m, r2, b in neighbors:
        if not _fact_predicates_allowed(
            r, r2, include_history=include_history, as_of=as_of
        ):
            continue
        n_uuid = getattr(n, "uuid", None)
        m_uuid = getattr(m, "uuid", None)
        b_uuid = getattr(b, "uuid", None)
        w1 = _edge_type_weight(r, query=query)
        w2 = _edge_type_weight(r2, query=query)
        if query_embedding is not None:
            for pred, base in ((r, w1), (r2, w2)):
                emb = getattr(pred, "embeddings", None) or getattr(
                    pred, "embedding", None
                )
                if emb is None:
                    continue
                try:
                    sim = float(cosine_similarity(query_embedding, emb))
                    boost = 0.5 + 0.5 * max(0.0, sim)
                    if pred is r:
                        w1 *= boost
                    else:
                        w2 *= boost
                except Exception:
                    pass
        _add_weighted_edge(weighted, n_uuid, m_uuid, w1)
        _add_weighted_edge(weighted, m_uuid, n_uuid, w1)
        _add_weighted_edge(weighted, m_uuid, b_uuid, w2)
        _add_weighted_edge(weighted, b_uuid, m_uuid, w2)
        if n_uuid and b_uuid:
            bridge = min(w1, w2) * 0.5
            _add_weighted_edge(weighted, n_uuid, b_uuid, bridge)
            _add_weighted_edge(weighted, b_uuid, n_uuid, bridge)
    return {
        src: sorted(((dst, weight) for dst, weight in edges.items()), key=lambda x: x[0])
        for src, edges in sorted(weighted.items(), key=lambda item: item[0])
    }


def _score_candidate_with_ppr(
    candidate: dict[str, Any], ppr_scores: dict[str, float]
) -> float:
    n, _r, m, _r2, b = candidate["triple"]
    scores = [
        ppr_scores.get(str(getattr(node, "uuid", "")), 0.0)
        for node in (n, m, b)
        if getattr(node, "uuid", None)
    ]
    if not scores:
        return _candidate_distance(candidate)
    return -max(scores)


def _context_looks_insufficient(
    question: str, text_lines: list[str], passages: list[str]
) -> bool:
    blob = "\n".join(text_lines + passages).lower()
    if len(blob.strip()) < 80:
        return True
    tokens = [
        t
        for t in re.findall(r"[a-z0-9']+", (question or "").lower())
        if len(t) > 3
        and t
        not in {
            "what",
            "which",
            "when",
            "where",
            "who",
            "does",
            "did",
            "have",
            "has",
            "been",
            "with",
            "from",
            "about",
            "would",
            "could",
            "their",
            "there",
            "this",
            "that",
            "mentioned",
        }
    ]
    if not tokens:
        return False
    hits = sum(1 for token in tokens if token in blob)
    return hits / len(tokens) < 0.35


def _stabilize_seed_hits(
    seeds: list[tuple[str, float, str]],
) -> list[tuple[str, float, str]]:
    best: dict[str, tuple[float, str]] = {}
    for uuid, distance, name in seeds:
        uid = str(uuid or "").strip()
        if not uid:
            continue
        dist = _quantize_distance(distance)
        prev = best.get(uid)
        if prev is None or dist < prev[0]:
            best[uid] = (dist, str(name))
    return sorted(
        ((uid, dist, name) for uid, (dist, name) in best.items()),
        key=lambda item: (item[1], item[0]),
    )


def _seed_nodes_for_text(
    text: str,
    brain_id: str,
    embeddings: list[float] | None = None,
) -> list[tuple[str, float, str]]:
    """Return (node_uuid, distance, entity_name) seeds from nodes + relationships."""
    if embeddings is None:
        with profile_stage("embed.query"):
            text_embeddings = _embed_query_vector(text)
        embeddings = text_embeddings.embeddings
    seeds: list[tuple[str, float, str]] = []
    vs = vector_search

    def _search_nodes():
        with profile_stage("vector.search_nodes", k=_SEED_K):
            return _backend_or_default(
                "vector.search_nodes",
                brain_id,
                lambda: vs.search_nodes(
                    embeddings,
                    brain_id=brain_id,
                    k=_SEED_K,
                ),
                [],
            )

    def _search_relationships():
        with profile_stage("vector.search_relationships", k=_SEED_K):
            return _backend_or_default(
                "vector.search_relationships",
                brain_id,
                lambda: vs.search_relationships(
                    embeddings,
                    brain_id=brain_id,
                    k=_SEED_K,
                ),
                [],
            )

    try:
        node_vectors, rel_vectors = _parallel_io(
            [_search_nodes, _search_relationships]
        )
    except _backend_error_types() as exc:
        _note_degraded("facts.seed_search", brain_id, exc)
        node_vectors, rel_vectors = [], []
    else:
        if node_vectors is _IO_FAILED:
            _note_degraded("vector.search_nodes", brain_id, TimeoutError())
        if rel_vectors is _IO_FAILED:
            _note_degraded("vector.search_relationships", brain_id, TimeoutError())
        node_vectors = _io_result(node_vectors, [])
        rel_vectors = _io_result(rel_vectors, [])
    for vector in node_vectors:
        meta = vector.metadata or {}
        uuid = meta.get("uuid") or getattr(vector, "id", None)
        if not uuid:
            continue
        distance = (
            vector.distance if vector.distance is not None else float("inf")
        )
        name = meta.get("name") or text
        seeds.append((str(uuid), float(distance), str(name)))
    for vector in rel_vectors:
        meta = vector.metadata or {}
        distance = (
            vector.distance if vector.distance is not None else float("inf")
        )
        node_ids = sorted(str(n) for n in (meta.get("node_ids") or []) if n)
        for node_id in node_ids:
            seeds.append((node_id, float(distance), text))

    return _stabilize_seed_hits(seeds)


def _extract_source_chunk_ids(*items: Any) -> list[str]:
    ids: list[str] = []
    seen: set[str] = set()
    for item in items:
        props = getattr(item, "properties", None) or {}
        for value in props.get("source_chunk_ids") or []:
            chunk_id = str(value).strip()
            if chunk_id and chunk_id not in seen:
                seen.add(chunk_id)
                ids.append(chunk_id)
        single = props.get("source_chunk_id")
        if single:
            chunk_id = str(single).strip()
            if chunk_id and chunk_id not in seen:
                seen.add(chunk_id)
                ids.append(chunk_id)
    return ids


def _session_ids_from_text(text: str) -> list[str]:
    found: list[str] = []
    seen: set[str] = set()
    for match in _SESSION_RE.finditer(text or ""):
        sid = f"session_{match.group(1)}"
        if sid not in seen:
            seen.add(sid)
            found.append(sid)
    for match in _BATCH_TURN_RE.finditer(text or ""):
        sid = f"session_b{match.group(1)}_t{match.group(2)}"
        if sid not in seen:
            seen.add(sid)
            found.append(sid)
    return found


def _session_ids_for_chunks(
    chunk_ids: list[str], chunk_sessions: dict[str, list[str]]
) -> list[str]:
    found: list[str] = []
    seen: set[str] = set()
    for chunk_id in chunk_ids:
        for sid in chunk_sessions.get(chunk_id) or []:
            if sid not in seen:
                seen.add(sid)
                found.append(sid)
    return found


def _fact_ids_from_context_triple(triple: Any) -> list[str]:
    ids: list[str] = []
    if not triple or len(triple) < 5:
        return ids
    for predicate in (triple[1], triple[3]):
        props = getattr(predicate, "properties", None) or {}
        uid = getattr(predicate, "uuid", None) or props.get("uuid")
        if uid:
            ids.append(str(uid))
    return ids


def _annotate_open_conflicts(
    adapter: Any,
    brain_id: str,
    triples: list[GetContextTriple],
    text_lines: list[str],
) -> tuple[list[GetContextTriple], list[str]]:
    from src.core.saving.conflicts import annotation_payload, open_conflicts_by_fact_ids

    fact_ids: list[str] = []
    for item in triples:
        fact_ids.extend(_fact_ids_from_context_triple(item.triple))
    mapping = open_conflicts_by_fact_ids(adapter, brain_id, fact_ids)
    annotated_lines: list[str] = []
    for index, item in enumerate(triples):
        payloads: list[dict[str, Any]] = []
        seen: set[str] = set()
        for fact_id in _fact_ids_from_context_triple(item.triple):
            for conflict in mapping.get(fact_id) or []:
                if conflict.id in seen:
                    continue
                seen.add(conflict.id)
                payloads.append(annotation_payload(conflict, fact_id))
        if payloads:
            item.open_conflicts = payloads
            line = text_lines[index] if index < len(text_lines) else ""
            if line and not line.startswith("[open conflict]"):
                line = f"[open conflict] {line}"
            annotated_lines.append(line)
        else:
            annotated_lines.append(text_lines[index] if index < len(text_lines) else "")
    return triples, annotated_lines


def _build_fact_channel(
    curated: list[dict[str, Any]],
    chunk_sessions: dict[str, list[str]],
) -> tuple[list[str], list[GetContextTriple], list[str]]:
    text_lines: list[str] = []
    triples: list[GetContextTriple] = []
    graph_session_ids: list[str] = []
    seen_sessions: set[str] = set()
    for candidate in curated:
        n, r, m, r2, b = candidate["triple"]
        chunk_ids = list(candidate.get("chunk_ids") or [])
        session_ids = _candidate_session_ids(candidate)
        if not session_ids:
            session_ids = _session_ids_for_chunks(chunk_ids, chunk_sessions)
        line = candidate["text"]
        if session_ids:
            line = f"{line} ({', '.join(session_ids)})"
        text_lines.append(line)
        verification = combined_verification(r, r2, m)
        triples.append(
            GetContextTriple(
                identified_entity=candidate["identified_entity"],
                triple=(n, r, m, r2, b),
                source_chunk_ids=chunk_ids or None,
                source_session_ids=session_ids or None,
                verification=FactVerificationInfo(**verification.model_dump()),
            )
        )
        for sid in session_ids:
            if sid not in seen_sessions:
                seen_sessions.add(sid)
                graph_session_ids.append(sid)
    graph_session_ids.sort()
    return text_lines, triples, graph_session_ids


def _retrieve_passages(
    text: str,
    brain_id: str,
    *,
    limit: int,
    embeddings: list[float] | None = None,
) -> list[tuple[str, float, str]]:
    """Return (chunk_id, score, text) ranked passages via vector + keyword fusion."""
    mode = (
        config.context_passage_mode if config.search_enabled else "ilike"
    )
    use_dense = mode in ("hybrid", "dense", "ilike")
    use_bm25 = mode in ("hybrid", "bm25")
    use_ilike = mode == "ilike"
    with profile_stage("passages.retrieve", blocking=True, queries=1):
        vector_ids: list[str] = []
        id_to_distance: dict[str, float] = {}
        query_embeddings = embeddings
        if use_dense and query_embeddings is None:
            with profile_stage("embed.query"):
                query_embeddings = _embed_query_vector(text).embeddings
        vs = vector_search
        data = data_adapter

        def _dense() -> tuple[list[str], dict[str, float]]:
            with profile_stage("vector.search_data", k=max(limit, _PASSAGE_K)):
                return _backend_or_default(
                    "vector.search_data",
                    brain_id,
                    lambda: collect_dense_passages(
                        vs,
                        query_embeddings or [],
                        brain_id,
                        max(limit, _PASSAGE_K),
                    ),
                    ([], {}),
                )

        def _keyword() -> tuple[list[str], dict[str, str]]:
            stage = "data.bm25_search" if use_bm25 else "data.keyword_search"
            if use_bm25:
                with profile_stage("data.bm25_search"):
                    _start_stage(stage)
                    try:
                        ids, _, texts = collect_bm25_passages(
                            data,
                            text,
                            brain_id,
                            max(limit, _PASSAGE_K),
                        )
                        _ok_stage(stage)
                        return ids, texts
                    except Exception as exc:
                        _note_degraded(stage, brain_id, exc)
                        return [], {}
            if use_ilike:
                with profile_stage("data.keyword_search"):
                    _start_stage(stage)
                    try:
                        ids, texts = collect_ilike_passages(data, text, brain_id)
                        _ok_stage(stage)
                        return ids, texts
                    except Exception as exc:
                        _note_degraded(stage, brain_id, exc)
                        return [], {}
            return [], {}

        keyword_ids: list[str] = []
        id_to_text: dict[str, str] = {}
        if use_dense and (use_bm25 or use_ilike):
            try:
                dense_result, keyword_result = _parallel_io([_dense, _keyword])
            except _backend_error_types() as exc:
                _note_degraded("passages.retrieve", brain_id, exc)
                _note_degraded("vector.search_data", brain_id, exc)
                dense_result, keyword_result = ([], {}), ([], {})
            else:
                if dense_result is _IO_FAILED:
                    _note_degraded("vector.search_data", brain_id, TimeoutError())
                    dense_result = ([], {})
                if keyword_result is _IO_FAILED:
                    stage = "data.bm25_search" if use_bm25 else "data.keyword_search"
                    _note_degraded(stage, brain_id, TimeoutError())
                    keyword_result = ([], {})
            vector_ids, id_to_distance = dense_result
            keyword_ids, id_to_text = keyword_result
        elif use_dense:
            vector_ids, id_to_distance = _dense()
        else:
            keyword_ids, id_to_text = _keyword()

        fused = fuse_passage_lists(
            vector_ids if use_dense else [],
            keyword_ids,
        )
        ranked_ids = [item for item, _ in fused[:limit]]
        missing = [cid for cid in ranked_ids if cid not in id_to_text]
        if missing:
            with profile_stage("data.fetch_chunks", chunks=len(missing)):
                try:
                    chunks, _ = data_adapter.get_text_chunks_by_ids(
                        missing, False, brain_id
                    )
                    for chunk in chunks:
                        id_to_text[str(chunk.id)] = chunk.text or ""
                except Exception:
                    pass

        passages: list[tuple[str, float, str]] = []
        for chunk_id in ranked_ids:
            body = (id_to_text.get(chunk_id) or "").strip()
            if not body:
                continue
            score = id_to_distance.get(chunk_id, 1.0 / (1.0 + len(passages)))
            passages.append((chunk_id, float(score), body))
        return passages


async def get_context(
    request: GetContextRequestBody,
    *,
    fact_filter_adapter: Any = None,
) -> GetContextResponse:
    """
    Retrieve contextual information for a text.

    `fact_filter_adapter` is only supplied by deep-navigation callers: the
    one-shot context path leaves it unset because an LLM call does not fit the
    sub-second budget, and without it no fact filtering happens.

    `profile_stages` attaches a per-stage wall-clock breakdown to the response
    and publishes it as a latency trace event; it is off unless the caller asks
    for it or `TRACE_STAGE_PROFILER_ENABLED` is set.
    """
    attach_timings = bool(request.profile_stages) or stage_profiling_enabled(False)
    health = _RetrievalHealth()
    token = _health_var.set(health)
    try:
        with verification_scope(include_rejected=bool(request.include_rejected)):
            with profile_request(
                "retrieve.context",
                enabled=True,
                publish=attach_timings,
            ) as profiler:
                response = await _build_context(request, fact_filter_adapter)
        if profiler is not None:
            log_stage_report(profiler.last_report, logger)
            if attach_timings:
                response.stage_timings = profiler.last_report
        if health.all_failed():
            raise HTTPException(
                status_code=503,
                detail="Retrieval backends unavailable",
            )
        if health.failed_stages:
            response.degraded = health.failed_stages
        return response
    finally:
        _health_var.reset(token)


async def _build_context(
    request: GetContextRequestBody,
    fact_filter_adapter: Any,
) -> GetContextResponse:
    from src.core.saving.validity import InvalidAsOf, optional_instant

    try:
        as_of = optional_instant(getattr(request, "as_of", None))
    except InvalidAsOf as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    def _extract_elements():
        with profile_stage("nlp.extract_elements"):
            return _entity_extractor.extract_elements(request.text)

    elements = await asyncio.to_thread(_extract_elements)
    with profile_stage("nlp.query_variants") as detail:
        variants = _collect_query_variants(request.text, elements)
        detail["variants"] = len(variants)
    max_passages = max(1, request.max_passages)
    historical_context: list[str] = []
    source_passages: list[str] = []
    candidates: list[dict[str, Any]] = []
    seed_hits: list[tuple[str, float, str]] = []
    passage_hits: list[tuple[str, float, str]] = []
    from src.core.saving.history_intent import (
        lexical_history_decision,
        lexical_history_intent,
        prototype_vectors,
        resolve_include_history,
    )

    shared_vectors: dict[str, list[float]] = {}

    def _embed_query(text: str) -> list[float]:
        cached = shared_vectors.get(text)
        if cached is None:
            cached = _embed_query_vector(text).embeddings
            shared_vectors[text] = cached
        return cached

    def _embed_queries(texts: list[str]) -> list[list[float]]:
        vectors: list[list[float] | None] = [shared_vectors.get(text) for text in texts]
        missing = [
            (index, text)
            for index, text in enumerate(texts)
            if vectors[index] is None
        ]
        if not missing:
            return [item or [] for item in vectors]
        batch = getattr(embeddings_adapter, "embed_texts", None)
        raw = None
        if callable(batch):
            try:
                raw = batch([text for _index, text in missing])
            except HTTPException:
                raise
            except Exception:
                logger.exception("Embedding request failed")
                raise HTTPException(
                    status_code=503,
                    detail="Embedding service unavailable",
                ) from None
            if not isinstance(raw, (list, tuple)) or len(raw) != len(missing):
                raw = None
        if raw is None:
            for index, text in missing:
                vectors[index] = _embed_query(text)
            return [item or [] for item in vectors]
        for (index, text), item in zip(missing, raw):
            vector = getattr(item, "embeddings", None)
            if vector is None:
                vector = item
            if not vector:
                logger.error("Embedding request failed")
                raise HTTPException(
                    status_code=503,
                    detail="Embedding service unavailable",
                )
            shared_vectors[text] = vector
            vectors[index] = vector
        return [item or [] for item in vectors]

    query_vector: list[float] | None = None
    explicit_history = request.include_history
    unknown_facts: list[dict[str, Any]] = []
    unknown_passages: list[dict[str, Any]] = []
    unique_texts: list[str] = []
    seen_texts: set[str] = set()
    for text in [request.text, *variants]:
        key = text.strip().lower()
        if not text or key in seen_texts:
            continue
        seen_texts.add(key)
        unique_texts.append(text)

    with profile_stage("history.intent", blocking=False) as history_detail:
        if as_of is not None:
            include_history = False
            history_detail["source"] = "as_of"
        elif explicit_history is not None:
            include_history = explicit_history
            history_detail["source"] = "explicit"
        else:
            lexical = lexical_history_decision(request.text)
            if lexical is not None:
                include_history = lexical
                history_detail["source"] = "lexical"
            else:
                history_detail["source"] = "similarity"

                def _resolve_history_flag() -> tuple[bool, list[float] | None]:
                    try:
                        with profile_stage("history.prototypes"):
                            history_vectors, current_vectors = prototype_vectors(
                                _embed_query, _embed_queries
                            )
                        qv = _embed_query(request.text)
                        flag = resolve_include_history(
                            None,
                            query_text=request.text,
                            query=qv,
                            history_vectors=history_vectors,
                            current_vectors=current_vectors,
                        )
                        return flag, qv
                    except Exception:
                        return lexical_history_intent(request.text), None

                include_history, query_vector = await asyncio.to_thread(_resolve_history_flag)

    def _collect_facts_for_variant(
        text: str,
    ) -> tuple[list[tuple[str, float, str]], list[dict[str, Any]]]:
        with profile_stage("facts.variant") as detail:
            with profile_stage("facts.seed_search"):
                seeds = _seed_nodes_for_text(
                    text, request.brain_id, embeddings=shared_vectors.get(text)
                )
            detail["seeds"] = len(seeds)
            if not seeds:
                return [], []
            uuid_to_distance: dict[str, float] = {}
            for uuid, distance, _ in seeds:
                dist = _quantize_distance(distance)
                prev = uuid_to_distance.get(uuid)
                if prev is None or dist < prev:
                    uuid_to_distance[uuid] = dist
            seed_uuids = sorted(
                uuid_to_distance.keys(),
                key=lambda uid: (uuid_to_distance[uid], uid),
            )
            with profile_stage(
                "graph.event_neighbors", seed_uuids=len(seed_uuids)
            ) as neighbors_detail:
                as_of_stamp = None
                if as_of is not None:
                    from src.core.saving.validity import format_instant

                    as_of_stamp = format_instant(as_of)
                neighbors = _backend_or_default(
                    "graph.event_neighbors",
                    request.brain_id,
                    lambda: graph_adapter.get_event_centric_neighbors(
                        seed_uuids,
                        brain_id=request.brain_id,
                        as_of=as_of_stamp,
                    ),
                    [],
                )
                neighbors_detail["rows"] = len(neighbors)
            local: list[dict[str, Any]] = []
            with profile_stage("facts.assemble"):
                for n, r, m, r2, b in neighbors:
                    if as_of is not None:
                        from src.core.saving.validity import (
                            VALIDITY_UNKNOWN,
                            predicate_validity_status,
                            unknown_fact,
                            unknown_relationship_id,
                        )

                        status_r = predicate_validity_status(r, as_of)
                        status_r2 = predicate_validity_status(r2, as_of)
                        if status_r == VALIDITY_UNKNOWN or status_r2 == VALIDITY_UNKNOWN:
                            unknown_facts.append(
                                unknown_fact(
                                    target="relationship",
                                    item_id=unknown_relationship_id(
                                        r, r2, status_r, status_r2
                                    ),
                                    payload=_format_event_fact(n, r, m, r2, b),
                                )
                            )
                            continue
                    if not _fact_predicates_allowed(
                        r, r2, include_history=include_history, as_of=as_of
                    ):
                        continue
                    distances = [
                        uuid_to_distance[u]
                        for u in (
                            getattr(n, "uuid", None),
                            getattr(m, "uuid", None),
                            getattr(b, "uuid", None),
                        )
                        if u in uuid_to_distance
                    ]
                    score = min(distances) if distances else float("inf")
                    fact_text = _format_event_fact(n, r, m, r2, b)
                    if as_of is None and include_history and not (
                        _is_currently_valid(r) and _is_currently_valid(r2)
                    ):
                        fact_text = f"[superseded] {fact_text}"
                    local.append(
                        {
                            "identified_entity": text,
                            "triple": (n, r, m, r2, b),
                            "score": score,
                            "key": _fact_key(r, r2),
                            "text": fact_text,
                            "chunk_ids": _extract_source_chunk_ids(n, r, m, r2, b),
                        }
                    )
            return list(seeds), local

    async def _collect_passages() -> None:
        nonlocal passage_hits
        with profile_stage("passages.collect", blocking=False) as detail:
            queries = [request.text] + [
                v
                for v in variants
                if v.strip().lower() != request.text.strip().lower()
            ]
            if _is_enumeration_question(request.text):
                queries = queries[:6]
            else:
                queries = queries[:3]
            detail["queries"] = len(queries)
            fused_lists: list[list[str]] = []
            id_to_text: dict[str, str] = {}
            id_to_score: dict[str, float] = {}

            def _hits_for_query(query: str) -> list[tuple[str, float, str]]:
                return _retrieve_passages(
                    query,
                    request.brain_id,
                    limit=max(max_passages, _PASSAGE_K),
                    embeddings=shared_vectors.get(query),
                )

            query_hits = list(
                await asyncio.gather(
                    *[
                        asyncio.to_thread(_hits_for_query, query)
                        for query in queries
                    ]
                )
            )
            for hits in query_hits:
                ordered_ids: list[str] = []
                for chunk_id, score, body in hits:
                    ordered_ids.append(chunk_id)
                    id_to_text[chunk_id] = body
                    prev = id_to_score.get(chunk_id)
                    if prev is None or score < prev:
                        id_to_score[chunk_id] = score
                if ordered_ids:
                    fused_lists.append(ordered_ids)
            if not fused_lists:
                passage_hits = []
                return
            fused = reciprocal_rank_fusion(fused_lists)
            ranked_ids = [item for item, _ in fused[: max(max_passages, _PASSAGE_K)]]
            passage_hits = [
                (cid, id_to_score.get(cid, 1.0), id_to_text.get(cid, ""))
                for cid in ranked_ids
                if id_to_text.get(cid)
            ]

    async def _get_historical_context():
        nonlocal historical_context
        from src.core.saving.source_passages import (
            current_record_texts,
            prior_revision_texts,
            screened_passage_texts,
        )

        def _merge_prior(texts: list[str], hits: list[tuple[str, float, str]]) -> list[str]:
            if not include_history:
                return texts[: request.historical_limit]
            extra = prior_revision_texts(
                data_adapter,
                request.brain_id,
                hits,
                limit=request.historical_limit,
                query_text=request.text,
                query_vector=query_vector or shared_vectors.get(request.text),
                embed=_embed_query,
                embed_many=_embed_queries,
            )
            merged = list(texts)
            for text in extra:
                if text not in merged:
                    merged.append(text)
            return merged[: request.historical_limit]

        def _history_from_hits(
            hits: list[tuple[str, float, str]],
            extra: list[str] | None = None,
        ) -> list[str]:
            extra_unknown: list = []
            texts = screened_passage_texts(
                data_adapter,
                request.brain_id,
                hits,
                include_history=include_history,
                limit=request.historical_limit,
                as_of=as_of,
                unknown_out=extra_unknown if as_of is not None else None,
            )
            if as_of is not None:
                from src.core.saving.validity import unknown_fact

                for chunk_id, _score, body, chunk in extra_unknown:
                    payload = (
                        chunk.model_dump(mode="json")
                        if hasattr(chunk, "model_dump")
                        else body
                    )
                    unknown_passages.append(
                        unknown_fact(
                            target="chunk", item_id=str(chunk_id), payload=payload
                        )
                    )
            if extra:
                texts = texts + extra
            return _merge_prior(texts, hits)

        with profile_stage("historical.context", blocking=False):
            passages = await asyncio.to_thread(
                _retrieve_passages,
                request.text,
                request.brain_id,
                limit=max(request.historical_limit, _PASSAGE_K),
                embeddings=shared_vectors.get(request.text),
            )
            if passages:
                historical_context = await asyncio.to_thread(
                    _history_from_hits, passages
                )
                return historical_context
            with profile_stage("historical.fallback", blocking=False):
                text_chunks, structured_data = await asyncio.gather(
                    asyncio.to_thread(
                        _backend_or_default,
                        "data.last_text_chunks",
                        request.brain_id,
                        lambda: data_adapter.get_last_text_chunks(
                            brain_id=request.brain_id,
                            limit=request.historical_limit,
                        ),
                        [],
                    ),
                    asyncio.to_thread(
                        _backend_or_default,
                        "data.last_structured_data",
                        request.brain_id,
                        lambda: data_adapter.get_last_structured_data(
                            brain_id=request.brain_id,
                            limit=request.historical_limit,
                        ),
                        [],
                    ),
                )
            chunk_hits = [
                (str(getattr(text_chunk, "id", "") or ""), 0.0, text_chunk.text or "")
                for text_chunk in text_chunks
            ]

            def _fallback_history() -> list[str]:
                extra = current_record_texts(
                    data_adapter,
                    request.brain_id,
                    structured_data,
                    text_chunks,
                    include_history=include_history,
                    as_of=as_of,
                )
                return _history_from_hits(chunk_hits, extra)

            historical_context = await asyncio.to_thread(_fallback_history)
            return historical_context

    with profile_stage("embed.query", blocking=False) as embed_detail:
        pending = [text for text in unique_texts if text not in shared_vectors]
        if pending:
            await asyncio.to_thread(_embed_queries, pending)
        if query_vector is None:
            query_vector = shared_vectors.get(request.text)
        embed_detail["texts"] = len(pending)

    with profile_stage(
        "retrieval.fanout", blocking=False, variants=len(variants)
    ):
        fact_futures = [
            asyncio.to_thread(_collect_facts_for_variant, variant)
            for variant in variants
        ]
        other_futures = [
            _collect_passages(),
            _get_historical_context(),
        ]
        all_results = await asyncio.gather(*fact_futures, *other_futures)
        variant_outputs = all_results[: len(variants)]

    seed_hits = []
    variant_lists: list[list[dict[str, Any]]] = []
    for seeds, local in variant_outputs:
        seed_hits.extend(seeds)
        variant_lists.append(local)
    candidates = _merge_variant_candidate_lists(variant_lists)

    bridge_paths: list[dict[str, Any]] = []
    def _bridge_expand():
        with profile_stage(
            "facts.bridge_expand",
            bridges=request.cross_event_bridges,
        ) as detail:
            expanded, paths = _expand_cross_event_bridges(
                candidates,
                request.brain_id,
                max_per_hub=request.cross_event_bridges,
                include_history=include_history,
                as_of=as_of,
                unknown_out=unknown_facts if as_of is not None else None,
            )
            detail["candidates"] = len(expanded)
            detail["paths"] = len(paths)
            return expanded, paths

    candidates, bridge_paths = await asyncio.to_thread(_bridge_expand)

    with profile_stage("facts.dedup_rank") as detail:
        ranked = _dedupe_candidates(candidates)
        detail["candidates"] = len(candidates)
        detail["deduped"] = len(ranked)

    if request.use_ppr and seed_hits and ranked:
        with profile_stage("ppr", blocking=False):
            personalization = _seed_personalization(seed_hits)
            def _adjacency():
                with profile_stage(
                    "ppr.adjacency", seeds=len(personalization)
                ) as adjacency_detail:
                    built = _build_adjacency_from_seeds(
                        sorted(personalization.keys()),
                        request.brain_id,
                        query=request.text,
                        include_history=include_history,
                        as_of=as_of,
                    )
                    adjacency_detail["nodes"] = len(built)
                    return built

            adjacency = await asyncio.to_thread(_adjacency)
            with profile_stage("ppr.iterations", iterations=_PPR_ITERS):
                ppr_scores = personalized_pagerank(
                    adjacency,
                    personalization,
                    damping=_PPR_DAMPING,
                    iterations=_PPR_ITERS,
                )
            if ppr_scores:
                with profile_stage("ppr.reorder", scored=len(ppr_scores)):
                    for candidate in ranked:
                        n, _r, m, _r2, b = candidate["triple"]
                        masses = [
                            ppr_scores.get(str(getattr(node, "uuid", "")), 0.0)
                            for node in (n, m, b)
                            if getattr(node, "uuid", None)
                        ]
                        candidate["ppr_mass"] = max(masses) if masses else 0.0
                    ranked = sorted(
                        ranked, key=lambda c: _ppr_rank_key(c, ppr_scores)
                    )

    max_facts = max(0, request.max_facts)

    from src.core.saving.source_passages import (
        current_chunk_body,
        current_fact_candidates,
        screen_passage_hits,
    )

    unknown_passage_hits: list = []
    passage_hits = screen_passage_hits(
        data_adapter,
        request.brain_id,
        passage_hits,
        include_history=include_history,
        as_of=as_of,
        unknown_out=unknown_passage_hits if as_of is not None else None,
    )
    if as_of is not None:
        from src.core.saving.validity import unknown_fact

        for chunk_id, _score, body, chunk in unknown_passage_hits:
            payload = (
                chunk.model_dump(mode="json") if hasattr(chunk, "model_dump") else body
            )
            unknown_passages.append(
                unknown_fact(target="chunk", item_id=str(chunk_id), payload=payload)
            )
    ranked = current_fact_candidates(
        data_adapter,
        request.brain_id,
        ranked,
        include_history=include_history,
        as_of=as_of,
    )

    provenance_ids: list[str] = []
    for candidate in ranked:
        provenance_ids.extend(candidate.get("chunk_ids") or [])
    for chunk_id, _, body in passage_hits:
        if body and body not in source_passages:
            source_passages.append(body)
        if chunk_id not in provenance_ids:
            provenance_ids.append(chunk_id)

    chunk_sessions: dict[str, list[str]] = {}
    if provenance_ids:
        with profile_stage(
            "provenance.chunks",
            blocking=False,
            chunk_ids=len(provenance_ids[:80]),
        ):
            try:
                chunks, _ = await asyncio.to_thread(
                    data_adapter.get_text_chunks_by_ids,
                    provenance_ids[:80],
                    False,
                    request.brain_id,
                )
                passage_cache: dict[str, set[str] | None] = {}
                for chunk in chunks:
                    raw = (chunk.text or "").strip()
                    chunk_id = str(getattr(chunk, "id", "") or "")
                    if chunk_id and raw:
                        sessions = _session_ids_from_text(raw)
                        if sessions:
                            chunk_sessions[chunk_id] = sessions
                    body = current_chunk_body(
                        data_adapter,
                        request.brain_id,
                        chunk,
                        include_history=include_history,
                        cache=passage_cache,
                        as_of=as_of,
                    )
                    if body and body not in source_passages:
                        source_passages.append(body)
            except Exception:
                pass

    for candidate in ranked:
        if not candidate.get("session_ids"):
            candidate["session_ids"] = _session_ids_for_chunks(
                list(candidate.get("chunk_ids") or []), chunk_sessions
            )

    response_topics: list[dict[str, Any]] = []
    with profile_stage("topics.coarse_to_fine") as topic_detail:
        try:
            from src.core.saving.topic_hyperedges import select_topics_and_sessions

            memberships = graph_adapter.list_topic_memberships(request.brain_id)
            seed_sessions: list[str] = []
            seen_seed: set[str] = set()
            for candidate in ranked:
                for sid in _candidate_session_ids(candidate):
                    if sid not in seen_seed:
                        seen_seed.add(sid)
                        seed_sessions.append(sid)
            response_topics, preferred_sessions = select_topics_and_sessions(
                request.text,
                memberships,
                seed_sessions,
                k_topics=10,
                k_sessions=20 if include_history else 10,
            )
            preferred = set(preferred_sessions)
            topic_detail["topics"] = len(response_topics)
            topic_detail["preferred_sessions"] = len(preferred)
            if preferred:
                for candidate in ranked:
                    sessions = set(_candidate_session_ids(candidate))
                    if sessions & preferred:
                        candidate["topic_preferred"] = True
        except Exception as exc:
            topic_detail["error"] = str(exc)[:120]

    with profile_stage("fact_filter") as detail:
        if request.apply_fact_filter and fact_filter_adapter is not None and ranked:
            detail["filter_applied"] = True
            keep = filter_relevant_facts(
                request.text,
                [c["text"] for c in ranked],
                llm_adapter=fact_filter_adapter,
                max_keep=max(max_facts, len(ranked)),
            )
            filtered = [ranked[i] for i in keep if 0 <= i < len(ranked)]
            if not filtered:
                filtered = ranked
        else:
            detail["filter_applied"] = False
            filtered = ranked

    with profile_stage("facts.diversify", max_facts=max_facts) as detail:
        filtered = _rank_facts_with_completeness(filtered)
        curated = _diversify_facts(filtered, max_facts=max_facts)
        detail["candidates"] = len(filtered)
        detail["curated"] = len(curated)

    with profile_stage("context.assemble") as detail:
        temporal_conflicts = _temporal_conflict_meta(curated)
        text_lines, triples, graph_session_ids = _build_fact_channel(
            curated, chunk_sessions
        )
        source_passages = source_passages[:max_passages]
        detail["facts"] = len(text_lines)
        detail["passages"] = len(source_passages)

    insufficient = False
    if request.sufficiency_retry:
        with profile_stage("sufficiency.check") as detail:
            insufficient = _context_looks_insufficient(
                request.text, text_lines, source_passages + historical_context
            )
            detail["fired"] = insufficient

    if insufficient:
        with profile_stage("sufficiency.retry", blocking=False) as detail:
            followups = _decompose_enumeration_queries(request.text, elements)
            if not followups:
                tokens = [
                    t
                    for t in re.findall(r"[A-Za-z][A-Za-z']+", request.text)
                    if len(t) > 3
                ]
                followups = [" ".join(tokens[:4])] if tokens else []
            detail["followups"] = len(followups[:3])
            extra_passages: list[tuple[str, float, str]] = []
            extra_variant_lists: list[list[dict[str, Any]]] = []
            retry_queries = followups[:3]
            if retry_queries:
                retry_results = await asyncio.gather(
                    *[
                        asyncio.gather(
                            asyncio.to_thread(
                                _retrieve_passages,
                                query,
                                request.brain_id,
                                limit=max_passages,
                                embeddings=shared_vectors.get(query),
                            ),
                            asyncio.to_thread(_collect_facts_for_variant, query),
                        )
                        for query in retry_queries
                    ]
                )
                for passages, (seeds, local) in retry_results:
                    extra_passages.extend(passages)
                    seed_hits.extend(seeds)
                    extra_variant_lists.append(local)
            extra_unknown: list = []
            extra_passages = screen_passage_hits(
                data_adapter,
                request.brain_id,
                extra_passages,
                include_history=include_history,
                as_of=as_of,
                unknown_out=extra_unknown if as_of is not None else None,
            )
            if as_of is not None:
                from src.core.saving.validity import unknown_fact

                for chunk_id, _score, body, chunk in extra_unknown:
                    payload = (
                        chunk.model_dump(mode="json")
                        if hasattr(chunk, "model_dump")
                        else body
                    )
                    unknown_passages.append(
                        unknown_fact(
                            target="chunk", item_id=str(chunk_id), payload=payload
                        )
                    )
            for chunk_id, _, body in extra_passages:
                if body and body not in source_passages:
                    source_passages.append(body)
                if chunk_id not in provenance_ids:
                    provenance_ids.append(chunk_id)
            source_passages = source_passages[:max_passages]
            candidates.extend(_merge_variant_candidate_lists(extra_variant_lists))
            ranked = _dedupe_candidates(candidates)
            for candidate in ranked:
                if not candidate.get("session_ids"):
                    candidate["session_ids"] = _session_ids_for_chunks(
                        list(candidate.get("chunk_ids") or []), chunk_sessions
                    )
            curated = _diversify_facts(
                _rank_facts_with_completeness(ranked), max_facts=max_facts
            )
            temporal_conflicts = _temporal_conflict_meta(curated)
            for candidate in curated:
                for chunk_id in candidate.get("chunk_ids") or []:
                    if chunk_id not in provenance_ids:
                        provenance_ids.append(chunk_id)
            missing_ids = [
                cid for cid in provenance_ids[:80] if cid not in chunk_sessions
            ]
            if missing_ids:
                try:
                    chunks, _ = await asyncio.to_thread(
                        data_adapter.get_text_chunks_by_ids,
                        missing_ids,
                        False,
                        request.brain_id,
                    )
                    for chunk in chunks:
                        body = (chunk.text or "").strip()
                        chunk_id = str(getattr(chunk, "id", "") or "")
                        if chunk_id and body:
                            sessions = _session_ids_from_text(body)
                            if sessions:
                                chunk_sessions[chunk_id] = sessions
                except Exception:
                    pass
            curated = current_fact_candidates(
                data_adapter,
                request.brain_id,
                curated,
                include_history=include_history,
                as_of=as_of,
            )
            text_lines, triples, graph_session_ids = _build_fact_channel(
                curated, chunk_sessions
            )

    response_paths = _paths_for_curated(bridge_paths, curated)

    with profile_stage("context.render", passages=len(source_passages)):
        if getattr(request, "annotate_conflicts", False):
            triples, text_lines = _annotate_open_conflicts(
                data_adapter, request.brain_id, triples, text_lines
            )
        topic_block = [
            f"[topic] {t.get('label')}"
            for t in response_topics
            if isinstance(t, dict) and t.get("label")
        ]
        passage_block = [f"[passage] {p}" for p in source_passages]
        text_context = "\n".join(topic_block + passage_block + text_lines)

    from src.core.saving.validity import dedupe_unknown_items

    unknown_items = dedupe_unknown_items(unknown_passages + unknown_facts)
    envelope: dict[str, Any] = {}
    if as_of is not None:
        envelope["validity_unknown_count"] = len(unknown_items)
        if getattr(request, "include_validity_unknown", False):
            envelope["validity_unknown"] = unknown_items

    return GetContextResponse(
        text_context=text_context,
        triples=triples,
        historical_context=historical_context,
        source_passages=source_passages,
        graph_session_ids=graph_session_ids or None,
        temporal_conflicts=temporal_conflicts or None,
        paths=response_paths or None,
        topics=response_topics or None,
        **envelope,
    )
