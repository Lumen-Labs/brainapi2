from __future__ import annotations

from datetime import datetime
from typing import Any, Iterable

from src.core.intent.models import IntentFact
from src.core.intent.parse import lexical_score, tokenize
from src.core.intent.status import intent_fact_from_chunk, intent_fact_from_row
from src.core.saving.validity import VALID_FROM, VALID_TO, SUPERSEDED_BY, is_valid_at
from src.core.verification.facts import verification_scope, verified_rank_boost

_QUERY_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "did",
        "do",
        "does",
        "for",
        "from",
        "how",
        "in",
        "is",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "was",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "whom",
        "with",
    }
)


def query_terms(text: str) -> list[str]:
    terms: list[str] = []
    seen: set[str] = set()
    for token in tokenize(text):
        variants = [token]
        if len(token) > 3 and token.endswith("s"):
            variants.append(token[:-1])
        for item in variants:
            if item in _QUERY_STOPWORDS or item in seen:
                continue
            seen.add(item)
            terms.append(item)
    return terms


def _search_rows(value: Any) -> list[Any]:
    if value is None:
        return []
    if hasattr(value, "results"):
        return list(value.results or [])
    if isinstance(value, (list, tuple)):
        return list(value)
    return []


def _properties_map(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        nested = value.get("properties")
        if isinstance(nested, dict):
            merged = dict(nested)
            merged.update({k: v for k, v in value.items() if k != "properties"})
            return merged
        return dict(value)
    props = getattr(value, "properties", None)
    return dict(props) if isinstance(props, dict) else {}


def _row_from_triple(triple: Any) -> dict[str, Any]:
    subject = getattr(triple, "subject", None)
    predicate = getattr(triple, "predicate", None)
    obj = getattr(triple, "object", None)
    if isinstance(triple, (list, tuple)) and len(triple) >= 3:
        subject, predicate, obj = triple[0], triple[1], triple[2]
    props = _properties_map(predicate)
    fact_id = str(
        getattr(predicate, "uuid", None)
        or props.get("uuid")
        or ""
    )
    props.setdefault("uuid", fact_id)
    return {
        "kind": "relationship",
        "uuid": fact_id,
        "name": str(getattr(predicate, "name", None) or props.get("name") or ""),
        "description": str(
            getattr(predicate, "description", None) or props.get("description") or ""
        ),
        "properties": props,
        "subject_uuid": str(getattr(subject, "uuid", None) or ""),
        "subject_name": str(getattr(subject, "name", None) or props.get("subject_name") or ""),
        "subject_labels": list(getattr(subject, "labels", None) or []),
        "object_uuid": str(getattr(obj, "uuid", None) or ""),
        "object_name": str(getattr(obj, "name", None) or props.get("object_name") or ""),
        "object_labels": list(getattr(obj, "labels", None) or []),
        "labels": [],
    }


def fact_in_window(fact: IntentFact, as_of: datetime | None) -> bool:
    fields = {
        VALID_FROM: fact.valid_at,
        VALID_TO: fact.invalid_at,
        "invalid_at": fact.invalid_at,
        SUPERSEDED_BY: fact.superseded_by,
    }
    return is_valid_at(fields, as_of, deprecated=False)


def _open_conflicts(data: Any, brain_id: str, fact_ids: list[str]) -> dict[str, list[str]]:
    getter = getattr(data, "get_open_conflicts_for_facts", None)
    if not callable(getter) or not fact_ids:
        return {}
    try:
        from src.core.saving.conflicts import open_conflicts_by_fact_ids

        mapping = open_conflicts_by_fact_ids(data, brain_id, fact_ids)
    except Exception:
        return {}
    out: dict[str, list[str]] = {}
    for fact_id, conflicts in mapping.items():
        others: list[str] = []
        for conflict in conflicts:
            for side in getattr(conflict, "sides", []) or []:
                other = getattr(side, "fact_id", None)
                if other and other != fact_id:
                    others.append(str(other))
        if others:
            out[fact_id] = list(dict.fromkeys(others))
    return out


def _annotate_conflicts(facts: list[IntentFact], data: Any, brain_id: str) -> None:
    if data is None:
        return
    contradicted = _open_conflicts(data, brain_id, [fact.fact_id for fact in facts if fact.fact_id])
    for fact in facts:
        extra = contradicted.get(fact.fact_id) or []
        if extra:
            fact.contradicted_by = list(dict.fromkeys([*fact.contradicted_by, *extra]))


def relevance_score(query: str, fact: IntentFact) -> float:
    blob = " ".join((fact.text, fact.subject, fact.object, fact.predicate, fact.natural_id()))
    score = lexical_score(query, blob)
    if query and query.lower() in (fact.text or "").lower():
        score = max(score, 1.0)
    return score


def rank_key(fact: IntentFact, query: str, distances: dict[str, float] | None = None) -> tuple:
    score = relevance_score(query, fact)
    boost = verified_rank_boost() if fact.status == "verified" else 0.0
    current = 0 if fact.status in {"verified", "unverified"} else 1
    distance = (distances or {}).get(fact.fact_id, 1.0)
    return (distance, -(score + boost), current, fact.fact_id)


def _keyword_queries(query: str) -> list[str]:
    terms = query_terms(query)
    ordered = [query.strip()] if query.strip() else []
    for term in terms:
        if term not in {item.lower() for item in ordered}:
            ordered.append(term)
    return ordered[:8]


def _keyword_graph_hits(
    graph: Any,
    brain_id: str,
    query: str,
    *,
    as_of: datetime | None,
) -> tuple[list[dict[str, Any]], list[str], dict[str, float]]:
    rows: dict[str, dict[str, Any]] = {}
    seeds: list[str] = []
    distances: dict[str, float] = {}
    search_rel = getattr(graph, "search_relationships", None)
    search_ent = getattr(graph, "search_entities", None)
    search_bm25 = getattr(graph, "search_nodes_bm25", None)
    for text in _keyword_queries(query):
        if callable(search_rel):
            try:
                found = search_rel(
                    brain_id=brain_id,
                    limit=40,
                    skip=0,
                    query_text=text,
                )
            except TypeError:
                try:
                    found = search_rel(brain_id, 40, 0, None, None, None, text)
                except Exception:
                    found = []
            except Exception:
                found = []
            for item in _search_rows(found):
                row = _row_from_triple(item) if not isinstance(item, dict) else item
                fact_id = str(row.get("uuid") or "")
                if not fact_id:
                    continue
                rows[fact_id] = row
                distances.setdefault(fact_id, 0.35)
                subject_uuid = str(row.get("subject_uuid") or "")
                object_uuid = str(row.get("object_uuid") or "")
                if subject_uuid:
                    seeds.append(subject_uuid)
                if object_uuid:
                    seeds.append(object_uuid)
        if callable(search_ent):
            try:
                found = search_ent(brain_id=brain_id, limit=20, skip=0, query_text=text)
            except TypeError:
                try:
                    found = search_ent(brain_id, 20, 0, None, text)
                except Exception:
                    found = []
            except Exception:
                found = []
            for node in _search_rows(found):
                uid = str(getattr(node, "uuid", None) or (node.get("uuid") if isinstance(node, dict) else "") or "")
                if uid:
                    seeds.append(uid)
        if callable(search_bm25) and text:
            try:
                hits = search_bm25(text, brain_id, limit=20) or []
            except Exception:
                hits = []
            for hit in hits:
                uid = str(getattr(hit, "uuid", None) or (hit.get("uuid") if isinstance(hit, dict) else "") or "")
                if uid:
                    seeds.append(uid)
    return list(rows.values()), list(dict.fromkeys(seeds)), distances


def _vector_graph_hits(
    query: str,
    brain_id: str,
    graph: Any,
    *,
    as_of: datetime | None,
) -> tuple[list[str], dict[str, float]]:
    seeds: list[str] = []
    distances: dict[str, float] = {}
    try:
        from src.services.api.controllers import retrieve as retrieve_mod

        hits = retrieve_mod._seed_nodes_for_text(query, brain_id)
    except Exception:
        return [], {}
    as_of_stamp = None
    if as_of is not None:
        from src.core.saving.validity import format_instant

        as_of_stamp = format_instant(as_of)
    for uuid, distance, _name in hits or []:
        if not uuid:
            continue
        seeds.append(str(uuid))
        prev = distances.get(str(uuid))
        dist = float(distance) if distance is not None else 1.0
        if prev is None or dist < prev:
            distances[str(uuid)] = dist
    expander = getattr(graph, "get_event_centric_neighbors", None)
    if callable(expander) and seeds:
        try:
            triples = expander(seeds, brain_id=brain_id, as_of=as_of_stamp)
        except TypeError:
            try:
                triples = expander(seeds, brain_id)
            except Exception:
                triples = []
        except Exception:
            triples = []
        for triple in triples or []:
            if isinstance(triple, (list, tuple)) and len(triple) >= 5:
                predicates = (triple[1], triple[3])
            else:
                predicates = (getattr(triple, "predicate", None),)
            for predicate in predicates:
                fact_id = str(getattr(predicate, "uuid", None) or "")
                if fact_id:
                    node_ids = [
                        str(getattr(item, "uuid", None) or "")
                        for item in (
                            triple[0] if isinstance(triple, (list, tuple)) else None,
                            triple[2] if isinstance(triple, (list, tuple)) and len(triple) > 2 else None,
                            triple[4] if isinstance(triple, (list, tuple)) and len(triple) > 4 else None,
                        )
                        if item is not None
                    ]
                    seed_distances = [distances[uid] for uid in node_ids if uid in distances]
                    distances.setdefault(
                        fact_id,
                        min(seed_distances) if seed_distances else 0.5,
                    )
    return list(dict.fromkeys(seeds)), distances


def _expand_neighbors(
    graph: Any,
    brain_id: str,
    seeds: Iterable[str],
) -> list[dict[str, Any]]:
    getter = getattr(graph, "get_neighbors", None)
    if not callable(getter):
        return []
    seed_list = [item for item in dict.fromkeys(seeds) if item]
    if not seed_list:
        return []
    try:
        neighbors = getter(seed_list, brain_id=brain_id)
    except TypeError:
        try:
            neighbors = getter(seed_list, brain_id)
        except Exception:
            return []
    except Exception:
        return []
    rows: list[dict[str, Any]] = []
    for subject_uuid, pairs in (neighbors or {}).items():
        for predicate, neighbor in pairs or []:
            direction = (getattr(predicate, "direction", None) or "").strip().lower()
            if direction == "in":
                continue
            props = _properties_map(predicate)
            fact_id = str(getattr(predicate, "uuid", None) or props.get("uuid") or "")
            if not fact_id:
                continue
            props.setdefault("uuid", fact_id)
            rows.append(
                {
                    "kind": "relationship",
                    "uuid": fact_id,
                    "name": str(getattr(predicate, "name", None) or ""),
                    "description": str(getattr(predicate, "description", None) or ""),
                    "properties": props,
                    "subject_uuid": str(subject_uuid),
                    "subject_name": str(props.get("subject_name") or ""),
                    "subject_labels": [],
                    "object_uuid": str(getattr(neighbor, "uuid", None) or ""),
                    "object_name": str(
                        getattr(neighbor, "name", None) or props.get("object_name") or ""
                    ),
                    "object_labels": list(getattr(neighbor, "labels", None) or []),
                    "labels": [],
                }
            )
    return rows


def _hydrate_rows(graph: Any, brain_id: str, rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    hydrated: list[dict[str, Any]] = []
    seen: set[str] = set()
    getter = getattr(graph, "get_fact_record", None)
    for row in rows:
        fact_id = str(row.get("uuid") or "")
        if not fact_id or fact_id in seen:
            continue
        seen.add(fact_id)
        full = None
        if callable(getter):
            try:
                full = getter(fact_id, brain_id)
            except TypeError:
                try:
                    full = getter(fact_id)
                except Exception:
                    full = None
            except Exception:
                full = None
        hydrated.append(full or row)
    return hydrated


def _passage_facts(
    data: Any,
    brain_id: str,
    query: str,
    *,
    include_rejected: bool,
    as_of: datetime | None,
) -> list[IntentFact]:
    facts: list[IntentFact] = []
    try:
        from src.services.api.controllers import retrieve as retrieve_mod

        hits = retrieve_mod._retrieve_passages(query, brain_id, limit=40)
    except Exception:
        hits = []
    getter = getattr(data, "get_text_chunks", None) if data is not None else None
    chunks_by_id: dict[str, Any] = {}
    if callable(getter):
        try:
            chunks, _total = getter(
                brain_id=brain_id,
                limit=80,
                skip=0,
                query_text=query or None,
            )
        except TypeError:
            try:
                chunks, _total = getter(brain_id=brain_id, limit=80, skip=0)
            except Exception:
                chunks = []
        except Exception:
            chunks = []
        for chunk in chunks or []:
            chunks_by_id[str(getattr(chunk, "id", "") or "")] = chunk
            fact = intent_fact_from_chunk(chunk, as_of=as_of)
            if fact.status == "rejected" and not include_rejected:
                continue
            facts.append(fact)
    for chunk_id, _score, body in hits or []:
        chunk = chunks_by_id.get(str(chunk_id))
        if chunk is None and data is not None:
            fact = IntentFact(
                fact_id=str(chunk_id),
                text=str(body or ""),
                status="unverified",
                kind="passage",
            )
            facts.append(fact)
            continue
        if chunk is None:
            continue
        fact = intent_fact_from_chunk(chunk, as_of=as_of)
        if fact.status == "rejected" and not include_rejected:
            continue
        facts.append(fact)
    return facts


def _merge_facts(*groups: Iterable[IntentFact]) -> list[IntentFact]:
    merged: dict[str, IntentFact] = {}
    for group in groups:
        for fact in group:
            key = fact.fact_id or f"{fact.kind}:{fact.natural_id()}"
            previous = merged.get(key)
            if previous is None or (previous.kind == "passage" and fact.kind != "passage"):
                merged[key] = fact
    return list(merged.values())


def collect_intent_facts(
    *,
    query: str,
    brain_id: str,
    graph: Any,
    data: Any | None = None,
    as_of: datetime | None = None,
    include_rejected: bool = False,
) -> tuple[list[IntentFact], dict[str, float]]:
    """Hybrid recall/check retrieval: keyword + vector + neighbor expand as available."""
    with verification_scope(include_rejected=True):
        keyword_rows, keyword_seeds, keyword_distances = _keyword_graph_hits(
            graph, brain_id, query, as_of=as_of
        )
        vector_seeds, vector_distances = _vector_graph_hits(
            query, brain_id, graph, as_of=as_of
        )
        neighbor_rows = _expand_neighbors(
            graph, brain_id, [*keyword_seeds, *vector_seeds]
        )
        rows = _hydrate_rows(graph, brain_id, [*keyword_rows, *neighbor_rows])
        seed_distances = {**{uid: 0.35 for uid in keyword_seeds}, **vector_distances}
        for row in rows:
            fact_id = str(row.get("uuid") or "")
            subject_uuid = str(row.get("subject_uuid") or "")
            object_uuid = str(row.get("object_uuid") or "")
            for uid in (subject_uuid, object_uuid, fact_id):
                if uid and uid in seed_distances:
                    keyword_distances.setdefault(fact_id, seed_distances[uid])
        graph_facts = [
            intent_fact_from_row(row, as_of=as_of)
            for row in rows
            if row
        ]
        passages = _passage_facts(
            data,
            brain_id,
            query,
            include_rejected=include_rejected,
            as_of=as_of,
        ) if data is not None else []
    merged = _merge_facts(graph_facts, passages)
    rejected_ids = {fact.fact_id for fact in merged if fact.status == "rejected"}
    filtered = [
        fact
        for fact in merged
        if (fact.fact_id not in rejected_ids or include_rejected)
        and (fact.status != "rejected" or include_rejected)
        and fact_in_window(fact, as_of)
    ]
    _annotate_conflicts(filtered, data, brain_id)
    distances = dict(keyword_distances)
    for fact in filtered:
        subject = fact.subject
        if fact.fact_id in vector_distances:
            distances[fact.fact_id] = vector_distances[fact.fact_id]
        else:
            for seed, dist in vector_distances.items():
                if seed and seed in {fact.fact_id}:
                    distances[fact.fact_id] = dist
    relevant = [
        fact
        for fact in filtered
        if relevance_score(query, fact) > 0 or fact.fact_id in distances
    ]
    relevant.sort(key=lambda fact: rank_key(fact, query, distances))
    return relevant, distances
