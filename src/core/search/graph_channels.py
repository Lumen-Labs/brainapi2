from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Iterable

from src.core.search.hybrid import dense_similarity, fuse_passage_lists, passage_snippet
from src.core.verification.facts import (
    entity_is_rejected,
    entity_is_verified,
    include_rejected_enabled,
    verification_extras,
    verified_rank_boost,
)
from src.utils.dates import parse_date_string, to_naive_utc

PASSAGES_CHANNEL = "passages"
ENTITIES_CHANNEL = "entities"
EVENTS_CHANNEL = "events"
COMMUNITIES_CHANNEL = "communities"
NEIGHBORS_CHANNEL = "neighbors"

CORE_CHANNELS = frozenset(
    {
        PASSAGES_CHANNEL,
        ENTITIES_CHANNEL,
        EVENTS_CHANNEL,
        COMMUNITIES_CHANNEL,
    }
)
GRAPH_CHANNELS = frozenset(
    {ENTITIES_CHANNEL, EVENTS_CHANNEL, COMMUNITIES_CHANNEL}
)

DEFAULT_COMMUNITY_LABELS = ("TYPE", "CLASS", "TOPIC")
DEFAULT_NEIGHBOR_FANOUT = 50
EVENT_LABEL = "EVENT"
ITEM_LABEL = "ENTITY"
HUB_ID_PREFIX = "hub:"
HUB_LABELS = frozenset({"ATTR", "TYPE", "CLASS", "TOPIC"})
EVENT_RECENCY_HALF_LIFE_DAYS = 365.0
_TOKEN_RE = re.compile(r"[A-Za-z0-9]{3,}")
_MAX_NAME_TOKENS = 6
_TOKEN_STOPWORDS = frozenset(
    {
        "the",
        "and",
        "for",
        "with",
        "from",
        "that",
        "this",
        "are",
        "was",
        "not",
        "without",
        "your",
        "you",
        "any",
        "all",
    }
)


@dataclass
class GraphHit:
    id: str
    channel: str
    score: float
    snippet: str
    labels: list[str] = field(default_factory=list)
    extras: dict[str, Any] | None = None


def parse_community_labels(raw: str | None) -> list[str]:
    items = [part.strip() for part in str(raw or "").split(",") if part.strip()]
    return items or list(DEFAULT_COMMUNITY_LABELS)


def selected_graph_channels(channels: list[str] | None) -> list[str]:
    selected: list[str] = []
    seen: set[str] = set()
    for item in channels or []:
        lowered = str(item or "").strip().lower()
        if lowered in GRAPH_CHANNELS and lowered not in seen:
            selected.append(lowered)
            seen.add(lowered)
    return selected


def _upper_set(values: Iterable[str] | None) -> set[str]:
    return {str(item).strip().upper() for item in (values or []) if str(item).strip()}


def _node_labels(node: Any) -> list[str]:
    raw = getattr(node, "labels", None) or []
    return [str(item) for item in raw if str(item).strip()]


def _node_uuid(node: Any) -> str:
    return str(getattr(node, "uuid", "") or "").strip()


def _node_name(node: Any) -> str:
    name = getattr(node, "name", None)
    if name:
        return str(name)
    return _node_uuid(node)


def _node_search_text(node: Any) -> str:
    props = getattr(node, "properties", None) or {}
    search_text = ""
    if isinstance(props, dict):
        search_text = str(props.get("search_text") or "").strip()
    description = str(getattr(node, "description", None) or "").strip()
    return search_text or compose_parts(_node_name(node), description)


def compose_parts(*parts: str) -> str:
    return " ".join(item for item in parts if item)


def _hub_kind(labels: Iterable[str] | None) -> str | None:
    have = _upper_set(labels)
    for kind in ("CLASS", "ATTR", "TYPE", "TOPIC"):
        if kind in have:
            return kind
    return None


def _happened_at(node: Any) -> str | None:
    raw = getattr(node, "happened_at", None)
    if raw:
        text = str(raw).strip()
        if text:
            return text
    props = getattr(node, "properties", None) or {}
    if isinstance(props, dict):
        value = props.get("happened_at")
        if value:
            text = str(value).strip()
            if text:
                return text
    return None


def _labels_match(node_labels: list[str], wanted: list[str] | None) -> bool:
    if not wanted:
        return True
    have = _upper_set(node_labels)
    need = _upper_set(wanted)
    return bool(have & need)


def is_item_entity(uuid: str, labels: Iterable[str] | None = None) -> bool:
    node_id = str(uuid or "").strip()
    if not node_id or node_id.startswith(HUB_ID_PREFIX):
        return False
    have = _upper_set(labels)
    if EVENT_LABEL in have and ITEM_LABEL not in have:
        return False
    if have & HUB_LABELS and ITEM_LABEL not in have:
        return False
    return True


def _entity_search_labels(node_labels: list[str] | None) -> list[str]:
    wanted = [str(item).strip() for item in (node_labels or []) if str(item).strip()]
    return wanted or [ITEM_LABEL]


def _name_match_score(name: str, query: str) -> float:
    text = (name or "").strip().lower()
    needle = (query or "").strip().lower()
    if not text:
        return 0.0
    score = 0.0
    if needle:
        if text == needle:
            score += 5.0
        elif needle in text:
            score += 3.0
    tokens = _query_tokens(query)
    if tokens:
        matched = [token for token in tokens if token in text]
        score += float(len(matched))
        if matched:
            score += len(matched) / len(tokens)
        for left, right in zip(tokens, tokens[1:]):
            if f"{left} {right}" in text:
                score += 0.75
    return score


def _search_tokens(query: str) -> list[str]:
    query_lower = (query or "").strip().lower()
    tokens: list[str] = []
    for token in _query_tokens(query):
        if token in _TOKEN_STOPWORDS or token == query_lower:
            continue
        tokens.append(token)
    return tokens


def _degree_idf(degree: int) -> float:
    return 1.0 / math.log2(2.0 + max(0, int(degree)))


def _parse_when(raw: str | None) -> datetime | None:
    if not raw:
        return None
    text = str(raw).strip()
    if not text:
        return None
    candidates = [text]
    if text.endswith("Z") and "+0000" not in text:
        candidates.append(text[:-1] + "+0000")
    for item in candidates:
        for fmt in (
            "%Y-%m-%dT%H:%M:%S%z",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%dT%H:%M:%S",
            "%m/%d/%Y",
            "%Y-%m-%d",
        ):
            try:
                parsed = datetime.strptime(item, fmt)
                return parsed
            except ValueError:
                continue
    return parse_date_string(text)


def recency_weight(
    happened_at: str | None,
    *,
    half_life_days: float = EVENT_RECENCY_HALF_LIFE_DAYS,
    now: datetime | None = None,
) -> float:
    if not happened_at or half_life_days <= 0:
        return 1.0
    parsed = _parse_when(happened_at)
    if parsed is None:
        return 1.0
    ref = now or datetime.now()
    age_days = max(
        0.0,
        (to_naive_utc(ref) - to_naive_utc(parsed)).total_seconds() / 86400.0,
    )
    return float(0.5 ** (age_days / float(half_life_days)))


def _query_tokens(query: str) -> list[str]:
    seen: set[str] = set()
    tokens: list[str] = []
    for match in _TOKEN_RE.finditer(query or ""):
        token = match.group(0).lower()
        if token in seen:
            continue
        seen.add(token)
        tokens.append(token)
        if len(tokens) >= _MAX_NAME_TOKENS:
            break
    return tokens


def _hit_from_node(
    node: Any,
    *,
    channel: str,
    score: float,
    extras: dict[str, Any] | None = None,
) -> GraphHit | None:
    node_id = _node_uuid(node)
    if not node_id:
        return None
    name = _node_name(node)
    payload = dict(extras or {})
    happened = _happened_at(node)
    if happened and "happened_at" not in payload:
        payload["happened_at"] = happened
    for key, value in verification_extras(node).items():
        payload.setdefault(key, value)
    return GraphHit(
        id=node_id,
        channel=channel,
        score=float(score),
        snippet=passage_snippet(name),
        labels=_node_labels(node),
        extras=payload or None,
    )


def _merge_hit(bucket: dict[str, GraphHit], hit: GraphHit | None) -> None:
    if hit is None or not hit.id:
        return
    current = bucket.get(hit.id)
    if current is None or hit.score > current.score:
        bucket[hit.id] = hit


def _ranked_hits(bucket: dict[str, GraphHit], k: int) -> list[GraphHit]:
    ordered = sorted(bucket.values(), key=lambda item: (-item.score, item.id))
    if k <= 0:
        return ordered
    return ordered[:k]


def _search_entities(
    graph: Any,
    *,
    brain_id: str,
    query_text: str | None,
    node_labels: list[str] | None,
    limit: int,
) -> list[Any]:
    if graph is None or limit <= 0:
        return []
    try:
        result = graph.search_entities(
            brain_id=brain_id,
            limit=limit,
            skip=0,
            node_labels=node_labels,
            query_text=query_text,
        )
    except Exception:
        return []
    raw = getattr(result, "results", None) or []
    if not isinstance(raw, (list, tuple)):
        return []
    return list(raw)


def _hydrate_nodes(graph: Any, uuids: list[str], brain_id: str) -> dict[str, Any]:
    if graph is None or not uuids:
        return {}
    unique = []
    seen: set[str] = set()
    for item in uuids:
        key = str(item or "").strip()
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(key)
    if not unique:
        return {}
    try:
        nodes = graph.get_by_uuids(unique, brain_id=brain_id)
    except Exception:
        nodes = []
        for key in unique:
            try:
                node = graph.get_by_uuid(key, brain_id=brain_id)
            except Exception:
                continue
            if node is not None:
                nodes.append(node)
    found: dict[str, Any] = {}
    for node in nodes or []:
        node_id = _node_uuid(node)
        if node_id:
            found[node_id] = node
    return found


def collect_entity_hits(
    *,
    query: str,
    brain_id: str,
    k: int,
    graph: Any,
    vector_search: Any = None,
    query_vector: list[float] | None = None,
    node_labels: list[str] | None = None,
    channel: str = ENTITIES_CHANNEL,
) -> list[GraphHit]:
    if k <= 0 or graph is None:
        return []
    bucket: dict[str, GraphHit] = {}
    wanted = _entity_search_labels(node_labels)
    restrict_items = not node_labels
    fetch_k = max(k * 8, 40)

    def _keep(node_id: str, labels: list[str]) -> bool:
        if not _labels_match(labels, wanted):
            return False
        if restrict_items and not is_item_entity(node_id, labels):
            return False
        return True

    def _add_node(node: Any, extra: float = 0.0) -> None:
        node_id = _node_uuid(node)
        labels = _node_labels(node)
        if not node_id or not _keep(node_id, labels):
            return
        if not include_rejected_enabled() and entity_is_rejected(node):
            return
        if entity_is_verified(node):
            extra += verified_rank_boost()
        score = _name_match_score(_node_search_text(node), query) + extra
        _merge_hit(
            bucket,
            _hit_from_node(node, channel=channel, score=score),
        )

    nodes = _search_entities(
        graph,
        brain_id=brain_id,
        query_text=query,
        node_labels=wanted,
        limit=fetch_k,
    )
    for node in nodes:
        _add_node(node)

    for token in _search_tokens(query):
        token_nodes = _search_entities(
            graph,
            brain_id=brain_id,
            query_text=token,
            node_labels=wanted,
            limit=fetch_k,
        )
        for node in token_nodes:
            _add_node(node)

    bm25_method = getattr(graph, "search_nodes_bm25", None)
    if callable(bm25_method):
        try:
            bm25_hits = bm25_method(
                query,
                brain_id,
                limit=fetch_k,
                node_labels=wanted,
            )
        except TypeError:
            try:
                bm25_hits = bm25_method(query, brain_id)
            except Exception:
                bm25_hits = []
        except Exception:
            bm25_hits = []
        for item in bm25_hits or []:
            node = item[0] if isinstance(item, (tuple, list)) and item else item
            score = 0.0
            if isinstance(item, (tuple, list)) and len(item) >= 2:
                try:
                    score = float(item[1])
                except (TypeError, ValueError):
                    score = 0.0
            extra = min(3.0, max(0.0, score))
            _add_node(node, extra=extra)

    if vector_search is not None and query_vector:
        try:
            vectors = vector_search.search_nodes(
                query_vector, brain_id=brain_id, k=fetch_k
            )
        except Exception:
            vectors = []
        ann_ids: list[str] = []
        ann_scores: dict[str, float] = {}
        ann_meta: dict[str, dict[str, Any]] = {}
        for vector in vectors or []:
            meta = getattr(vector, "metadata", None) or {}
            node_id = str(meta.get("uuid") or getattr(vector, "id", "") or "").strip()
            if not node_id:
                continue
            ann_ids.append(node_id)
            distance = getattr(vector, "distance", None)
            ann_scores[node_id] = dense_similarity(
                float(distance) if distance is not None else float("inf")
            )
            if isinstance(meta, dict):
                ann_meta[node_id] = meta
        hydrated = _hydrate_nodes(graph, ann_ids, brain_id)
        for node_id in ann_ids:
            node = hydrated.get(node_id)
            labels = _node_labels(node) if node is not None else []
            extra = 0.5 * float(ann_scores.get(node_id, 0.0))
            if node is None:
                meta = ann_meta.get(node_id) or {}
                raw_labels = meta.get("labels") or []
                if isinstance(raw_labels, str):
                    raw_labels = [raw_labels]
                labels = [str(item) for item in raw_labels if str(item).strip()]
                if not labels and meta.get("type"):
                    labels = [str(meta.get("type"))]
                if not _keep(node_id, labels):
                    continue
                name = str(meta.get("name") or node_id)
                _merge_hit(
                    bucket,
                    GraphHit(
                        id=node_id,
                        channel=channel,
                        score=_name_match_score(name, query) + extra,
                        snippet=passage_snippet(name),
                        labels=labels,
                        extras=None,
                    ),
                )
                continue
            _add_node(node, extra=extra)

    return _ranked_hits(bucket, k)


def collect_event_hits(
    *,
    query: str,
    brain_id: str,
    k: int,
    graph: Any,
    vector_search: Any = None,
    query_vector: list[float] | None = None,
) -> list[GraphHit]:
    hits = collect_entity_hits(
        query=query,
        brain_id=brain_id,
        k=k,
        graph=graph,
        vector_search=vector_search,
        query_vector=query_vector,
        node_labels=[EVENT_LABEL],
        channel=EVENTS_CHANNEL,
    )
    scored: list[GraphHit] = []
    for hit in hits:
        extras = dict(hit.extras or {})
        if not include_rejected_enabled() and extras.get("verification_state") == "rejected":
            continue
        happened = extras.get("happened_at")
        weight = recency_weight(str(happened) if happened else None)
        if happened:
            extras["recency"] = weight
        score = float(hit.score) * weight
        if extras.get("verification_state") == "verified":
            score += verified_rank_boost()
        scored.append(
            GraphHit(
                id=hit.id,
                channel=EVENTS_CHANNEL,
                score=score,
                snippet=hit.snippet,
                labels=list(hit.labels or []),
                extras=extras or None,
            )
        )
    return _ranked_hits({item.id: item for item in scored}, k)


def _event_centric_others(
    graph: Any,
    uuid: str,
    brain_id: str,
    *,
    skip_labels: set[str],
    fanout: int,
) -> list[Any]:
    found: list[Any] = []
    seen = {uuid}
    try:
        triples = graph.get_event_centric_neighbors([uuid], brain_id=brain_id)
    except Exception:
        triples = []
    for triple in triples or []:
        if not isinstance(triple, (tuple, list)) or len(triple) < 5:
            continue
        n, _p1, m, _p2, b = triple[:5]
        for node in (n, m, b):
            node_id = _node_uuid(node)
            if not node_id or node_id in seen:
                continue
            labels = _upper_set(_node_labels(node))
            if skip_labels and labels & skip_labels:
                continue
            seen.add(node_id)
            found.append(node)
            if len(found) >= fanout:
                return found
    return found


def _direct_others(
    graph: Any,
    uuid: str,
    brain_id: str,
    *,
    skip_labels: set[str],
    fanout: int,
    skip_event_wrappers: bool,
) -> list[Any]:
    found: list[Any] = []
    seen = {uuid}
    try:
        neighbors = graph.get_neighbors([uuid], brain_id=brain_id)
    except Exception:
        return found
    rows = []
    if isinstance(neighbors, dict):
        rows = neighbors.get(uuid) or []
    for entry in rows:
        node = entry[1] if isinstance(entry, (tuple, list)) and len(entry) >= 2 else entry
        node_id = _node_uuid(node)
        if not node_id or node_id in seen:
            continue
        labels = _upper_set(_node_labels(node))
        if skip_event_wrappers and EVENT_LABEL in labels:
            try:
                inner = graph.get_neighbors([node_id], brain_id=brain_id)
            except Exception:
                inner = {}
            inner_rows = []
            if isinstance(inner, dict):
                inner_rows = inner.get(node_id) or []
            for inner_entry in inner_rows:
                tip = (
                    inner_entry[1]
                    if isinstance(inner_entry, (tuple, list)) and len(inner_entry) >= 2
                    else inner_entry
                )
                tip_id = _node_uuid(tip)
                if not tip_id or tip_id in seen:
                    continue
                tip_labels = _upper_set(_node_labels(tip))
                if skip_labels and tip_labels & skip_labels:
                    continue
                if EVENT_LABEL in tip_labels:
                    continue
                seen.add(tip_id)
                found.append(tip)
                if len(found) >= fanout:
                    return found
            continue
        if skip_labels and labels & skip_labels:
            continue
        seen.add(node_id)
        found.append(node)
        if len(found) >= fanout:
            return found
    return found


def adjacent_nodes(
    graph: Any,
    uuid: str,
    brain_id: str,
    *,
    skip_labels: Iterable[str] | None = None,
    fanout: int = DEFAULT_NEIGHBOR_FANOUT,
    skip_event_wrappers: bool = True,
) -> list[Any]:
    if graph is None or not uuid or fanout <= 0:
        return []
    skip = _upper_set(skip_labels)
    found = _event_centric_others(
        graph, uuid, brain_id, skip_labels=skip, fanout=fanout
    )
    if len(found) >= fanout:
        return found[:fanout]
    seen = {uuid, *(_node_uuid(node) for node in found)}
    for node in _direct_others(
        graph,
        uuid,
        brain_id,
        skip_labels=skip,
        fanout=fanout,
        skip_event_wrappers=skip_event_wrappers,
    ):
        node_id = _node_uuid(node)
        if not node_id or node_id in seen:
            continue
        seen.add(node_id)
        found.append(node)
        if len(found) >= fanout:
            break
    return found[:fanout]


def _member_hits_for_hub(
    *,
    hub: GraphHit,
    graph: Any,
    brain_id: str,
    query: str,
    skip: set[str],
    fanout: int,
) -> tuple[list[GraphHit], dict[str, Any]]:
    members = adjacent_nodes(
        graph,
        hub.id,
        brain_id,
        skip_labels=skip,
        fanout=fanout,
        skip_event_wrappers=True,
    )
    degree = max(len(members), 1)
    idf = _degree_idf(degree)
    scored: list[GraphHit] = []
    nodes: dict[str, Any] = {}
    for member in members:
        labels_of = _node_labels(member)
        member_id = _node_uuid(member)
        if not is_item_entity(member_id, labels_of):
            continue
        lexical = _name_match_score(_node_search_text(member), query)
        hit = _hit_from_node(
            member,
            channel=COMMUNITIES_CHANNEL,
            score=float(hub.score) * idf * (1.0 + lexical),
            extras={"hub_id": hub.id, "hub_name": hub.snippet},
        )
        if hit is None:
            continue
        scored.append(hit)
        nodes[member_id] = member
    scored.sort(key=lambda item: (-item.score, item.id))
    return scored, nodes


def _hybrid_rank_community_hits(
    hits: list[GraphHit],
    *,
    query: str,
    brain_id: str,
    k: int,
    graph: Any,
    vector_search: Any = None,
    query_vector: list[float] | None = None,
) -> list[GraphHit]:
    if not hits:
        return []
    by_id = {hit.id: hit for hit in hits}
    candidate_ids = list(by_id.keys())
    lexical_ids: list[str] = []
    bm25_method = getattr(graph, "search_nodes_bm25", None)
    if callable(bm25_method):
        try:
            bm25_hits = bm25_method(
                query,
                brain_id,
                limit=max(k * 8, len(candidate_ids)),
                node_uuids=candidate_ids,
            )
        except Exception:
            bm25_hits = []
        for item in bm25_hits or []:
            node = item[0] if isinstance(item, (tuple, list)) and item else item
            node_id = _node_uuid(node)
            if node_id in by_id and node_id not in lexical_ids:
                lexical_ids.append(node_id)
    dense_ids: list[str] = []
    if vector_search is not None and query_vector:
        try:
            vectors = vector_search.search_nodes(
                query_vector, brain_id=brain_id, k=max(k * 8, len(candidate_ids))
            )
        except Exception:
            vectors = []
        for vector in vectors or []:
            meta = getattr(vector, "metadata", None) or {}
            node_id = str(meta.get("uuid") or getattr(vector, "id", "") or "").strip()
            if node_id in by_id and node_id not in dense_ids:
                dense_ids.append(node_id)
    fused = fuse_passage_lists(dense_ids, lexical_ids)
    if not fused:
        return _ranked_hits(by_id, k)
    bucket: dict[str, GraphHit] = {}
    for node_id, score in fused:
        hit = by_id.get(node_id)
        if hit is None:
            continue
        bucket[node_id] = GraphHit(
            id=hit.id,
            channel=hit.channel,
            score=float(score),
            snippet=hit.snippet,
            labels=list(hit.labels or []),
            extras=dict(hit.extras or {}) or None,
        )
    for hit in hits:
        if hit.id not in bucket:
            bucket[hit.id] = GraphHit(
                id=hit.id,
                channel=hit.channel,
                score=0.0,
                snippet=hit.snippet,
                labels=list(hit.labels or []),
                extras=dict(hit.extras or {}) or None,
            )
    return _ranked_hits(bucket, k)


def collect_community_hits(
    *,
    query: str,
    brain_id: str,
    k: int,
    graph: Any,
    vector_search: Any = None,
    query_vector: list[float] | None = None,
    community_labels: list[str] | None = None,
    fanout: int = DEFAULT_NEIGHBOR_FANOUT,
) -> list[GraphHit]:
    labels = [item for item in (community_labels or []) if str(item).strip()]
    if not labels:
        labels = list(DEFAULT_COMMUNITY_LABELS)
    hub_labels = list(labels)
    if "ATTR" not in _upper_set(hub_labels):
        hub_labels.append("ATTR")
    hubs = collect_entity_hits(
        query=query,
        brain_id=brain_id,
        k=k,
        graph=graph,
        vector_search=vector_search,
        query_vector=query_vector,
        node_labels=hub_labels,
        channel=COMMUNITIES_CHANNEL,
    )
    skip = _upper_set(hub_labels) | HUB_LABELS | {EVENT_LABEL}
    cap = max(1, int(fanout))
    hub_cap = max(1, min(len(hubs), 8))
    kinds = {
        kind
        for kind in (_hub_kind(hub.labels) for hub in hubs[:hub_cap])
        if kind
    }
    expand_cap = max(cap, 200) if len(kinds) >= 2 else cap
    members_by_kind: dict[str, set[str]] = {}
    union_bucket: dict[str, GraphHit] = {}
    for hub in hubs[:hub_cap]:
        kind = _hub_kind(hub.labels) or "ATTR"
        scored, _nodes = _member_hits_for_hub(
            hub=hub,
            graph=graph,
            brain_id=brain_id,
            query=query,
            skip=skip,
            fanout=expand_cap,
        )
        kind_ids = members_by_kind.setdefault(kind, set())
        for hit in scored[:expand_cap]:
            kind_ids.add(hit.id)
            _merge_hit(union_bucket, hit)
    kind_sets = [ids for ids in members_by_kind.values() if ids]
    selected_ids: set[str] | None = None
    if len(kind_sets) >= 2:
        selected_ids = set(kind_sets[0])
        for ids in kind_sets[1:]:
            selected_ids &= ids
        if not selected_ids:
            selected_ids = None
    if selected_ids is None:
        candidates = list(union_bucket.values())
    else:
        candidates = [
            hit for hit in union_bucket.values() if hit.id in selected_ids
        ]
    return _hybrid_rank_community_hits(
        candidates,
        query=query,
        brain_id=brain_id,
        k=k,
        graph=graph,
        vector_search=vector_search,
        query_vector=query_vector,
    )


def expand_neighbor_hits(
    seeds: list[GraphHit],
    *,
    brain_id: str,
    k: int,
    graph: Any,
    community_labels: list[str] | None = None,
    fanout: int = DEFAULT_NEIGHBOR_FANOUT,
) -> list[GraphHit]:
    if not seeds or graph is None or k <= 0:
        return []
    skip = _upper_set(community_labels)
    seed_ids = {hit.id for hit in seeds}
    bucket: dict[str, GraphHit] = {}
    cap = max(1, int(fanout))
    for seed in seeds:
        skip_events = seed.channel != EVENTS_CHANNEL
        members = adjacent_nodes(
            graph,
            seed.id,
            brain_id,
            skip_labels=skip,
            fanout=cap,
            skip_event_wrappers=skip_events,
        )
        degree = max(len(members), 1)
        idf = _degree_idf(degree)
        scored: list[GraphHit] = []
        for member in members:
            node_id = _node_uuid(member)
            if not node_id or node_id in seed_ids:
                continue
            hit = _hit_from_node(
                member,
                channel=NEIGHBORS_CHANNEL,
                score=float(seed.score) * idf,
                extras={"seed_id": seed.id, "seed_channel": seed.channel},
            )
            if hit is not None:
                scored.append(hit)
        scored.sort(key=lambda item: (-item.score, item.id))
        for hit in scored[:cap]:
            _merge_hit(bucket, hit)
    return _ranked_hits(bucket, k)
