import asyncio
from typing import Any, Optional

from fastapi import HTTPException

from src.config import config
from src.core.search.catalog_graph import node_id_from_passage_text
from src.core.search.personalize import personalize_ranked_ids
from src.core.search.graph_channels import (
    COMMUNITIES_CHANNEL,
    ENTITIES_CHANNEL,
    EVENTS_CHANNEL,
    GraphHit,
    NEIGHBORS_CHANNEL,
    PASSAGES_CHANNEL,
    collect_community_hits,
    collect_entity_hits,
    collect_event_hits,
    expand_neighbor_hits,
    is_item_entity,
    selected_graph_channels,
)
from src.core.search.hooks import (
    SearchPluginError,
    rerank_max_k_for_mode,
    resolve_reranker,
    resolve_retrievers,
    retrieve_k_for_mode,
)
from src.core.search.hybrid import (
    collect_bm25_passages,
    collect_dense_passages,
    collect_literal_residual,
    dense_similarity,
    extras_from_metadata,
    facet_counts_from_extras,
    frozen_head_merge,
    fuse_passage_lists,
    hit_matches_extras,
    merge_hit_extras,
    passage_snippet,
)
from src.lib.tracing.profiler import profile_request, profile_stage
from src.services.api.constants.requests import (
    SearchHit,
    SearchHitScores,
    SearchRequestBody,
    SearchResponse,
)
from src.services.data.main import data_adapter
from src.services.kg_agent.main import embeddings_adapter, graph_adapter, vector_store_adapter
from src.utils.vector_search import VectorSearchFacade
from src.core.verification.facts import verification_scope

vector_search = VectorSearchFacade(vector_store_adapter)


def _require_search_enabled() -> None:
    if not config.search_enabled:
        raise HTTPException(
            status_code=404,
            detail="Search is disabled. Set SEARCH_ENABLED=true to enable /retrieve/search.",
        )


def _plugin_http(exc: SearchPluginError) -> HTTPException:
    return HTTPException(status_code=400, detail=str(exc))


def _fetch_snippets(ids: list[str], known: dict[str, str], brain_id: str) -> dict[str, str]:
    texts, _ = _fetch_chunk_fields(ids, known, {}, brain_id)
    return texts


def _fetch_chunk_fields(
    ids: list[str],
    known_texts: dict[str, str],
    known_extras: dict[str, dict[str, Any]],
    brain_id: str,
) -> tuple[dict[str, str], dict[str, dict[str, Any]]]:
    texts = dict(known_texts)
    extras: dict[str, dict[str, Any]] = dict(known_extras)
    missing = [
        chunk_id
        for chunk_id in ids
        if chunk_id not in texts or chunk_id not in extras
    ]
    if not missing:
        return texts, extras
    try:
        chunks, _ = data_adapter.get_text_chunks_by_ids(missing, False, brain_id)
    except Exception:
        return texts, extras
    if not isinstance(chunks, (list, tuple)):
        return texts, extras
    for chunk in chunks:
        chunk_id = str(getattr(chunk, "id", "") or "")
        if not chunk_id:
            continue
        if chunk_id not in texts:
            texts[chunk_id] = getattr(chunk, "text", "") or ""
        parsed = extras_from_metadata(getattr(chunk, "metadata", None))
        if parsed:
            extras[chunk_id] = merge_hit_extras(extras.get(chunk_id), parsed) or parsed
        elif chunk_id not in extras:
            extras[chunk_id] = {}
    return texts, extras


def _run_dense(
    query_vector: list[float],
    brain_id: str,
    k: int,
    extras_out: dict[str, dict[str, str]] | None = None,
):
    with profile_stage("search.dense"):
        return collect_dense_passages(
            vector_search, query_vector, brain_id, k, extras_out=extras_out
        )


def _run_bm25(
    query: str,
    brain_id: str,
    k: int,
    extras_out: dict[str, dict[str, str]] | None = None,
):
    with profile_stage("search.bm25"):
        return collect_bm25_passages(
            data_adapter, query, brain_id, k, extras_out=extras_out
        )


def _want_passages(channels: list[str] | None) -> bool:
    if not channels:
        return True
    return any(str(item).strip().lower() == PASSAGES_CHANNEL for item in channels)


def _collect_graph_hits(
    request: SearchRequestBody,
    *,
    query_vector: list[float] | None,
    graph_channels: list[str],
) -> list[GraphHit]:
    k = request.k
    brain_id = request.brain_id
    fanout = int(getattr(config, "search_neighbor_fanout", 50) or 50)
    community_labels = request.community_labels
    if community_labels is None:
        community_labels = list(config.search_community_labels)
    hits: list[GraphHit] = []
    if ENTITIES_CHANNEL in graph_channels:
        hits.extend(
            collect_entity_hits(
                query=request.query,
                brain_id=brain_id,
                k=k,
                graph=graph_adapter,
                vector_search=vector_search,
                query_vector=query_vector,
                node_labels=request.node_labels,
            )
        )
    if EVENTS_CHANNEL in graph_channels:
        hits.extend(
            collect_event_hits(
                query=request.query,
                brain_id=brain_id,
                k=k,
                graph=graph_adapter,
                vector_search=vector_search,
                query_vector=query_vector,
            )
        )
    if COMMUNITIES_CHANNEL in graph_channels:
        hits.extend(
            collect_community_hits(
                query=request.query,
                brain_id=brain_id,
                k=k,
                graph=graph_adapter,
                vector_search=vector_search,
                query_vector=query_vector,
                community_labels=community_labels,
                fanout=fanout,
            )
        )
    if request.expand == "neighbors" and hits:
        hits.extend(
            expand_neighbor_hits(
                hits,
                brain_id=brain_id,
                k=k,
                graph=graph_adapter,
                community_labels=community_labels,
                fanout=fanout,
            )
        )
    return hits


async def search(request: SearchRequestBody) -> SearchResponse:
    _require_search_enabled()
    with verification_scope(include_rejected=bool(request.include_rejected)):
        with profile_request(
            "retrieve.search", enabled=request.profile_stages
        ) as profiler:
            response = await _search(request)
    if profiler is not None:
        response.stage_timings = profiler.last_report
    return response


async def _search(request: SearchRequestBody) -> SearchResponse:
    from src.core.saving.source_passages import screen_passage_hits
    from src.core.saving.validity import InvalidAsOf, optional_instant

    try:
        as_of = optional_instant(getattr(request, "as_of", None))
    except InvalidAsOf as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    try:
        reranker = resolve_reranker(request.rerank)
        plugin_retrievers = resolve_retrievers(request.channels)
    except SearchPluginError as exc:
        raise _plugin_http(exc) from exc

    k = request.k
    k_ret = retrieve_k_for_mode(request.mode, k)
    if as_of is not None:
        k_ret = min(200, max(k_ret, k * 8))
    rerank_max = rerank_max_k_for_mode(request.mode)
    brain_id = request.brain_id
    fusion = request.fusion or config.search_fusion
    fusion_alpha = (
        request.fusion_alpha
        if request.fusion_alpha is not None
        else config.search_fusion_alpha
    )
    want_passages = _want_passages(request.channels)
    graph_channels = selected_graph_channels(request.channels)
    want_graph = bool(graph_channels)
    use_dense = want_passages and config.search_use_dense
    use_bm25 = want_passages and config.search_use_bm25
    query_vector: Optional[list[float]] = None
    if use_dense or want_graph:
        with profile_stage("embed.query", blocking=False):
            embedded = await asyncio.to_thread(
                embeddings_adapter.embed_text, request.query
            )
            query_vector = list(embedded.embeddings or [])

    dense_extras: dict[str, dict[str, str]] = {}
    bm25_extras: dict[str, dict[str, str]] = {}
    retrieve_jobs = []
    if use_dense:
        retrieve_jobs.append(
            asyncio.to_thread(
                _run_dense, query_vector or [], brain_id, k_ret, dense_extras
            )
        )
    if use_bm25:
        retrieve_jobs.append(
            asyncio.to_thread(
                _run_bm25, request.query, brain_id, k_ret, bm25_extras
            )
        )

    dense_ids: list[str] = []
    dense_distances: dict[str, float] = {}
    bm25_ids: list[str] = []
    bm25_scores: dict[str, float] = {}
    texts: dict[str, str] = {}
    plugin_id_lists: list[list[str]] = []
    plugin_scores: dict[str, dict[str, float]] = {}
    plugin_channels: dict[str, str] = {}
    graph_hits: list[GraphHit] = []
    literal_ids: list[str] = []
    with profile_stage("search.retrieve", blocking=False):
        gathered = await asyncio.gather(*retrieve_jobs) if retrieve_jobs else []
        offset = 0
        if use_dense:
            dense_ids, dense_distances = gathered[offset]
            offset += 1
        if use_bm25:
            bm25_ids, bm25_scores, texts = gathered[offset]
        for name, retriever in plugin_retrievers:
            with profile_stage(f"search.plugin.{name}"):
                ids, scores, plugin_texts = await asyncio.to_thread(
                    retriever, request.query, brain_id, k_ret
                )
            plugin_id_lists.append(ids)
            plugin_scores[name] = scores
            texts.update(plugin_texts)
            for item_id in ids:
                plugin_channels.setdefault(str(item_id), f"plugin:{name}")
        if want_graph:
            try:
                graph_hits = await asyncio.to_thread(
                    _collect_graph_hits,
                    request,
                    query_vector=query_vector,
                    graph_channels=graph_channels,
                )
            except Exception:
                graph_hits = []
        if getattr(config, "search_literal_fill", False):
            lit_ids, literal_texts = collect_literal_residual(
                data_adapter, request.query, brain_id, k_ret
            )
            literal_ids = list(lit_ids)
            texts.update(literal_texts)
            for item_id in literal_ids:
                plugin_channels.setdefault(str(item_id), PASSAGES_CHANNEL)

    passage_extras = {**dense_extras, **bm25_extras}
    dense_sims = {
        chunk_id: dense_similarity(distance)
        for chunk_id, distance in dense_distances.items()
    }
    extra_id_lists: list[list[str]] = []
    graph_by_channel: dict[str, list[str]] = {}
    graph_by_id: dict[str, GraphHit] = {}
    for hit in graph_hits:
        graph_by_channel.setdefault(hit.channel, []).append(hit.id)
        current = graph_by_id.get(hit.id)
        if current is None or hit.score > current.score:
            graph_by_id[hit.id] = hit

    def _item_ids(channel_name: str) -> list[str]:
        ids = graph_by_channel.get(channel_name) or []
        if channel_name != EVENTS_CHANNEL:
            ids = [
                item_id
                for item_id in ids
                if is_item_entity(
                    item_id,
                    graph_by_id[item_id].labels if item_id in graph_by_id else [],
                )
            ]
        return ids

    graph_channel_ids = {
        ENTITIES_CHANNEL: _item_ids(ENTITIES_CHANNEL),
        EVENTS_CHANNEL: _item_ids(EVENTS_CHANNEL),
        COMMUNITIES_CHANNEL: _item_ids(COMMUNITIES_CHANNEL),
        NEIGHBORS_CHANNEL: _item_ids(NEIGHBORS_CHANNEL),
    }
    for ids in graph_channel_ids.values():
        if ids:
            extra_id_lists.append(ids)
            for item_id in ids:
                texts.setdefault(item_id, graph_by_id[item_id].snippet)
    fused = fuse_passage_lists(
        dense_ids if use_dense else [],
        bm25_ids if use_bm25 else [],
        fusion=fusion,
        alpha=fusion_alpha,
        dense_similarities=dense_sims,
        bm25_scores=bm25_scores,
        extra_id_lists=extra_id_lists,
    )
    ranked_ids = [item for item, _ in fused]
    sidecar_lists = [ids for ids in plugin_id_lists if ids]
    if literal_ids:
        sidecar_lists.append(literal_ids)
    if sidecar_lists:
        ranked_ids = frozen_head_merge(
            ranked_ids,
            sidecar_lists,
            head_k=10,
            k=k_ret,
        )
    else:
        ranked_ids = ranked_ids[:k_ret]
    rerank_scores: dict[str, float] = {}
    if reranker is not None and ranked_ids:
        head_n = min(rerank_max, len(ranked_ids))
        head_ids = ranked_ids[:head_n]
        tail_ids = ranked_ids[head_n:]
        with profile_stage("search.rerank", blocking=False, chunks=head_n):
            texts = await asyncio.to_thread(
                _fetch_snippets, head_ids, texts, brain_id
            )
            candidates = [
                {
                    "id": chunk_id,
                    "text": texts.get(chunk_id) or "",
                    "score": float(dict(fused).get(chunk_id, 0.0)),
                }
                for chunk_id in head_ids
            ]
            reranked = await asyncio.to_thread(
                reranker, request.query, candidates, head_n
            )
        ordered: list[str] = []
        seen: set[str] = set()
        for item in reranked or []:
            chunk_id = str((item or {}).get("id") or "")
            if not chunk_id or chunk_id not in head_ids or chunk_id in seen:
                continue
            ordered.append(chunk_id)
            seen.add(chunk_id)
            if item.get("score") is not None:
                rerank_scores[chunk_id] = float(item["score"])
        for chunk_id in head_ids:
            if chunk_id not in seen:
                ordered.append(chunk_id)
        ranked_ids = ordered + tail_ids
    ranked_ids = ranked_ids[:k_ret]
    fused_map = dict(fused)

    extras_by_id: dict[str, dict[str, Any]] = dict(passage_extras)
    for item_id, graph_hit in graph_by_id.items():
        extras_by_id[item_id] = (
            merge_hit_extras(extras_by_id.get(item_id), graph_hit.extras) or {}
        )

    with profile_stage("search.snippets", blocking=False, chunks=len(ranked_ids)):
        texts, extras_by_id = await asyncio.to_thread(
            _fetch_chunk_fields, ranked_ids, texts, extras_by_id, brain_id
        )

    if request.extras:
        ranked_ids = [
            item_id
            for item_id in ranked_ids
            if hit_matches_extras(extras_by_id.get(item_id), request.extras)
        ]
    if as_of is not None:
        passage_ids = [
            item_id for item_id in ranked_ids if item_id not in graph_by_id
        ]
        if passage_ids:
            screened = screen_passage_hits(
                data_adapter,
                brain_id,
                [
                    (item_id, 0.0, texts.get(item_id) or "")
                    for item_id in passage_ids
                ],
                include_history=False,
                as_of=as_of,
            )
            allowed = {item_id for item_id, _, _ in screened}
            ranked_ids = [
                item_id
                for item_id in ranked_ids
                if item_id in graph_by_id or item_id in allowed
            ]
            ranked_ids = ranked_ids[: retrieve_k_for_mode(request.mode, k)]
    node_id_by_hit: dict[str, str | None] = {}
    for item_id in ranked_ids:
        graph_hit = graph_by_id.get(item_id)
        if graph_hit is not None and is_item_entity(graph_hit.id, graph_hit.labels):
            node_id_by_hit[item_id] = graph_hit.id
        else:
            node_id_by_hit[item_id] = node_id_from_passage_text(
                texts.get(item_id) or ""
            )
    personalize_scores: dict[str, float] = {}
    if request.target:
        retrieve_for_blend = {
            item_id: float(
                rerank_scores.get(item_id, fused_map.get(item_id, 0.0))
            )
            for item_id in ranked_ids
        }
        ranked_ids, personalize_scores = personalize_ranked_ids(
            query=request.query,
            ranked_ids=ranked_ids,
            retrieve_scores=retrieve_for_blend,
            node_id_by_hit=node_id_by_hit,
            target=request.target,
            brain_id=brain_id,
        )
    ranked_ids = ranked_ids[:k]

    channel_of: dict[str, str] = {}
    for chunk_id in list(dense_ids) + list(bm25_ids):
        channel_of.setdefault(chunk_id, PASSAGES_CHANNEL)
    for hit in graph_hits:
        channel_of.setdefault(hit.id, hit.channel)
    for chunk_id, plugin_channel in plugin_channels.items():
        channel_of.setdefault(chunk_id, plugin_channel)

    hits: list[SearchHit] = []
    for chunk_id in ranked_ids:
        body = texts.get(chunk_id) or ""
        rrf = fused_map.get(chunk_id)
        plugin = {
            name: scores[chunk_id]
            for name, scores in plugin_scores.items()
            if chunk_id in scores
        }
        graph_hit = graph_by_id.get(chunk_id)
        extras = merge_hit_extras(extras_by_id.get(chunk_id), graph_hit.extras if graph_hit else None)
        scores = SearchHitScores(
            bm25=bm25_scores.get(chunk_id),
            dense=dense_sims.get(chunk_id),
            rrf=rrf if fusion == "rrf" else None,
            cc=rrf if fusion == "cc" else None,
            rerank=rerank_scores.get(chunk_id),
            plugin=plugin or None,
            graph=graph_hit.score if graph_hit is not None else None,
            personalize=personalize_scores.get(chunk_id),
        )
        node_id = node_id_by_hit.get(chunk_id)
        hits.append(
            SearchHit(
                id=chunk_id,
                channel=channel_of.get(chunk_id, PASSAGES_CHANNEL),
                score=float(
                    personalize_scores.get(
                        chunk_id,
                        rerank_scores.get(chunk_id, fused_map.get(chunk_id, 0.0)),
                    )
                ),
                scores=scores,
                snippet=passage_snippet(body),
                labels=list(graph_hit.labels) if graph_hit is not None else [],
                extras=extras,
                node_id=node_id,
            )
        )
    node_ids: list[str] = []
    seen_nodes: set[str] = set()
    for hit in hits:
        if not hit.node_id or hit.node_id in seen_nodes:
            continue
        seen_nodes.add(hit.node_id)
        node_ids.append(hit.node_id)
    return SearchResponse(
        hits=hits,
        facets=facet_counts_from_extras([hit.extras for hit in hits]),
        channel_lists={
            "dense": list(dense_ids),
            "bm25": list(bm25_ids),
            "entities": list(graph_channel_ids[ENTITIES_CHANNEL]),
            "events": list(graph_channel_ids[EVENTS_CHANNEL]),
            "communities": list(graph_channel_ids[COMMUNITIES_CHANNEL]),
            "neighbors": list(graph_channel_ids[NEIGHBORS_CHANNEL]),
            "literal": list(literal_ids),
        },
        node_ids=node_ids,
    )
