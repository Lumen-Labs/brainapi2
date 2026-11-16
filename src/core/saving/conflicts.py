"""Detect and store cross-source conflicts between graph facts.

Deterministic matching is the default. An optional batched call through the
configured small LLM judges leftover semantic pairs. Ingest-time detection
is gated by ``CONFLICT_DETECT_ON_INGEST`` (default off).
"""

from __future__ import annotations

import hashlib
import math
import os
import re
from typing import Any, Callable, Iterable

from src.constants.conflicts import (
    ConflictEntity,
    ConflictResolution,
    ConflictSide,
    ConflictSource,
    FactConflict,
    conflict_from_document,
    conflict_to_document,
)
from src.core.saving.source_passages import source_key_from_meta
from src.core.saving.validity import (
    SUPERSEDED_BY,
    VALID_FROM,
    VALID_TO,
    format_instant,
    parse_instant,
    utc_now,
)

_REL_PAGE = 200
_MAX_SCAN = 10000
_LLM_BATCH = 10

MULTI_VALUED_PREDICATES = frozenset(
    {
        "MADE",
        "TARGETED",
        "OCCURRED_WITHIN",
        "OCCURRED",
        "HUB_BRIDGE",
        "RELATED_TO",
        "HAS_TAG",
        "TAGS",
        "INCLUDES",
        "CONTAINS",
        "MEMBER_OF",
        "PART_OF",
        "WORKS_WITH",
        "KNOWS",
        "MENTIONS",
        "CITES",
        "ATTENDED",
        "PARTICIPATED_IN",
        "HAS_ITEM",
        "HAS_FEATURE",
        "HAS",
        "INITIATED",
        "PERFORMED",
        "EXPERIENCED",
        "COVERED",
        "AFFECTED",
        "RESULTED",
        "WITHIN",
    }
)

FUNCTIONAL_PREDICATES = frozenset(
    {
        "HAS_WINDOW",
        "REFUND_WINDOW",
        "RETURNS_WITHIN",
        "OFFERS_REFUND_WITHIN",
        "REQUIRES",
        "DURATION",
        "PERIOD",
        "LIMIT",
        "THRESHOLD",
        "IS",
        "HAS_STATUS",
        "STATUS",
        "POLICY",
        "HAS_POLICY",
        "RATE",
        "AMOUNT",
        "PRICE",
        "DEADLINE",
        "VALIDITY",
        "RETENTION",
        "WARRANTY",
        "EQUALS",
        "STATES",
        "SPECIFIES",
        "APPLIES",
        "ALLOWS",
        "PROHIBITS",
        "HEADQUARTERED_IN",
        "LOCATED_IN",
    }
)

_DURATION_UNITS = {
    "day": 1.0,
    "days": 1.0,
    "d": 1.0,
    "week": 7.0,
    "weeks": 7.0,
    "wk": 7.0,
    "wks": 7.0,
    "fortnight": 14.0,
    "month": 30.0,
    "months": 30.0,
    "mo": 30.0,
    "year": 365.0,
    "years": 365.0,
    "yr": 365.0,
    "yrs": 365.0,
    "hour": 1.0 / 24.0,
    "hours": 1.0 / 24.0,
    "hr": 1.0 / 24.0,
    "hrs": 1.0 / 24.0,
}

_NUMBER_RE = re.compile(
    r"(?P<num>\d+(?:\.\d+)?)\s*(?P<unit>days?|d|weeks?|wks?|wk|months?|mo|years?|yrs?|yr|hours?|hrs?|hr|fortnight)?",
    re.I,
)
_WORD_NUMBERS = {
    "a": 1.0,
    "an": 1.0,
    "one": 1.0,
    "two": 2.0,
    "three": 3.0,
    "four": 4.0,
    "five": 5.0,
    "six": 6.0,
    "seven": 7.0,
    "eight": 8.0,
    "nine": 9.0,
    "ten": 10.0,
    "fourteen": 14.0,
    "thirty": 30.0,
    "sixty": 60.0,
}
_WORD_DURATION_RE = re.compile(
    r"\b(?P<word>a|an|one|two|three|four|five|six|seven|eight|nine|ten|fourteen|thirty|sixty)"
    r"\s+(?P<unit>days?|weeks?|months?|years?|fortnight)\b",
    re.I,
)
_ANTONYM_PAIRS = (
    ("refundable", "non-refundable"),
    ("refundable", "nonrefundable"),
    ("allowed", "prohibited"),
    ("allowed", "forbidden"),
    ("permitted", "prohibited"),
    ("yes", "no"),
    ("true", "false"),
    ("included", "excluded"),
    ("applies", "does not apply"),
    ("required", "optional"),
)
_NEGATION_RE = re.compile(
    r"\b(not|no|never|non|without|cannot|can't)\b", re.I
)


def conflict_detect_on_ingest_enabled() -> bool:
    raw = os.getenv("CONFLICT_DETECT_ON_INGEST")
    if raw is None:
        from src.config import config

        return bool(getattr(config, "conflict_detect_on_ingest", False))
    return raw.strip().lower() in {"1", "true", "yes"}


def conflict_detect_llm_enabled() -> bool:
    raw = os.getenv("CONFLICT_DETECT_LLM")
    if raw is None:
        from src.config import config

        return bool(getattr(config, "conflict_detect_llm", False))
    return raw.strip().lower() in {"1", "true", "yes"}


def conflict_detect_llm_max_pairs() -> int:
    raw = os.getenv("CONFLICT_DETECT_LLM_MAX_PAIRS")
    if raw:
        try:
            return max(0, int(raw))
        except ValueError:
            return 20
    from src.config import config

    return int(getattr(config, "conflict_detect_llm_max_pairs", 20) or 20)


def normalize_predicate(name: str | None) -> str:
    return re.sub(r"[^A-Z0-9]+", "_", (name or "").upper()).strip("_")


def claim_key_for(subject_uuid: str, predicate: str) -> str:
    raw = f"{subject_uuid}|{normalize_predicate(predicate)}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def pair_key_for(fact_a: str, fact_b: str) -> str:
    left, right = sorted((str(fact_a), str(fact_b)))
    return f"{left}|{right}"


def conflict_id_for(pair_key: str) -> str:
    return hashlib.sha256(pair_key.encode("utf-8")).hexdigest()[:24]


def _props(value: Any) -> dict[str, Any]:
    props = getattr(value, "properties", None)
    if isinstance(props, dict):
        nested = props.get("properties")
        if isinstance(nested, dict):
            merged = dict(nested)
            merged.update({k: v for k, v in props.items() if k != "properties"})
            return merged
        return props
    if isinstance(value, dict):
        return value
    return {}


def _chunk_ids(props: dict[str, Any]) -> list[str]:
    raw = props.get("source_chunk_ids") or props.get("source_chunk_id") or []
    if isinstance(raw, str):
        raw = [raw]
    ids: list[str] = []
    seen: set[str] = set()
    for item in raw:
        text = str(item).strip()
        if text and text not in seen:
            seen.add(text)
            ids.append(text)
    return ids


def _verification_state_of(predicate: Any, props: dict[str, Any]) -> str:
    from src.core.saving.conflict_verification import read_fact_verification_state

    return (
        read_fact_verification_state(
            str(getattr(predicate, "uuid", None) or props.get("uuid") or ""),
            entity=predicate if predicate is not None else props,
        )
        or "unverified"
    )


def _polarity_of(node: Any, props: dict[str, Any]) -> str:
    node_props = getattr(node, "properties", None) if node is not None else None
    if not isinstance(node_props, dict):
        node_props = {}
    value = (
        node_props.get("polarity")
        or props.get("polarity")
        or getattr(node, "polarity", None)
    )
    text = str(value or "neutral").strip().lower()
    if text in {"positive", "negative", "neutral"}:
        return text
    return "neutral"


def _amount_of(predicate: Any, props: dict[str, Any], value_text: str) -> float | None:
    raw = getattr(predicate, "amount", None)
    if raw is None:
        raw = props.get("amount")
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        return float(raw)
    parsed = parse_quantity(value_text)
    if parsed is None:
        return None
    return parsed[0]


def parse_quantity(text: str | None) -> tuple[float, str] | None:
    if not text:
        return None
    lowered = text.strip().lower()
    word = _WORD_DURATION_RE.search(lowered)
    if word:
        number = _WORD_NUMBERS.get(word.group("word").lower())
        unit = word.group("unit").lower()
        days = _DURATION_UNITS.get(unit)
        if number is not None and days is not None:
            return number * days, "days"
    match = _NUMBER_RE.search(lowered)
    if not match:
        return None
    number = float(match.group("num"))
    unit = (match.group("unit") or "").lower()
    if unit in _DURATION_UNITS:
        return number * _DURATION_UNITS[unit], "days"
    if unit:
        return number, unit
    return number, "number"


def normalize_value(text: str | None) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def values_equivalent(left: str, right: str) -> bool:
    a = normalize_value(left)
    b = normalize_value(right)
    if a == b:
        return True
    qa = parse_quantity(a)
    qb = parse_quantity(b)
    if qa and qb and qa[1] == qb[1]:
        return math.isclose(qa[0], qb[0], rel_tol=0.02, abs_tol=0.05)
    return False


def opposite_polarity(left: str, right: str) -> bool:
    return {left, right} == {"positive", "negative"}


def antonym_or_negation(left: str, right: str) -> bool:
    a = normalize_value(left)
    b = normalize_value(right)
    for first, second in _ANTONYM_PAIRS:
        if (first in a and second in b) or (second in a and first in b):
            return True
    a_neg = bool(_NEGATION_RE.search(a))
    b_neg = bool(_NEGATION_RE.search(b))
    return a_neg != b_neg and bool(a) and bool(b) and a.replace("not ", "") != b.replace("not ", "")


def is_functional_predicate(predicate: str, values: Iterable[str] | None = None) -> bool:
    name = normalize_predicate(predicate)
    if name in MULTI_VALUED_PREDICATES:
        return False
    if name in FUNCTIONAL_PREDICATES:
        return True
    samples = [normalize_value(v) for v in (values or []) if v]
    if samples and all(parse_quantity(v) is not None for v in samples):
        return True
    return False


def intervals_overlap(left: dict[str, Any], right: dict[str, Any]) -> bool:
    start_a = parse_instant(left.get(VALID_FROM) or left.get("valid_at"))
    start_b = parse_instant(right.get(VALID_FROM) or right.get("valid_at"))
    end_a = parse_instant(
        left.get(VALID_TO) or left.get("invalid_at") or left.get("superseded_at")
    )
    end_b = parse_instant(
        right.get(VALID_TO) or right.get("invalid_at") or right.get("superseded_at")
    )
    begin_a = start_a.timestamp() if start_a else float("-inf")
    begin_b = start_b.timestamp() if start_b else float("-inf")
    finish_a = end_a.timestamp() if end_a else float("inf")
    finish_b = end_b.timestamp() if end_b else float("inf")
    return begin_a < finish_b and begin_b < finish_a


def same_source_revision(left: dict[str, Any], right: dict[str, Any]) -> bool:
    key_a = left.get("source_key")
    key_b = right.get("source_key")
    if key_a and key_b and key_a == key_b:
        return True
    id_a = left.get("fact_id")
    id_b = right.get("fact_id")
    if id_a and right.get(SUPERSEDED_BY) == id_a:
        return True
    if id_b and left.get(SUPERSEDED_BY) == id_b:
        return True
    return False


def snapshot_from_triple(triple: Any) -> dict[str, Any] | None:
    if not triple or len(triple) < 3:
        return None
    subject, predicate, obj = triple[0], triple[1], triple[2]
    props = _props(predicate)
    fact_id = str(
        getattr(predicate, "uuid", None) or props.get("uuid") or ""
    ).strip()
    if not fact_id:
        return None
    subject_uuid = str(getattr(subject, "uuid", "") or "")
    predicate_name = str(getattr(predicate, "name", "") or props.get("rel_type") or "")
    object_name = str(getattr(obj, "name", "") or "")
    description = str(getattr(predicate, "description", "") or props.get("description") or "")
    value = object_name or description
    amount = _amount_of(predicate, props, value)
    return {
        "fact_id": fact_id,
        "subject_uuid": subject_uuid,
        "subject_name": str(getattr(subject, "name", "") or ""),
        "subject_labels": list(getattr(subject, "labels", None) or []),
        "predicate": predicate_name,
        "object_uuid": str(getattr(obj, "uuid", "") or ""),
        "object_name": object_name,
        "object_labels": list(getattr(obj, "labels", None) or []),
        "value": value,
        "description": description,
        "polarity": _polarity_of(obj, props),
        "verification_state": _verification_state_of(predicate, props),
        "amount": amount,
        VALID_FROM: props.get(VALID_FROM) or props.get("valid_at"),
        VALID_TO: props.get(VALID_TO) or props.get("invalid_at"),
        SUPERSEDED_BY: props.get(SUPERSEDED_BY),
        "deprecated": bool(getattr(predicate, "deprecated", False) or props.get("deprecated")),
        "source_chunk_ids": _chunk_ids(props),
        "source_timestamp": props.get("source_timestamp"),
        "source_key": props.get("source_key"),
        "document": props.get("document") or props.get("source"),
    }


def classify_disagreement(left: dict[str, Any], right: dict[str, Any]) -> tuple[str, float] | None:
    if values_equivalent(left.get("value") or "", right.get("value") or ""):
        if opposite_polarity(left.get("polarity") or "neutral", right.get("polarity") or "neutral"):
            return "polarity_conflict", 0.99
        amount_a = left.get("amount")
        amount_b = right.get("amount")
        if (
            isinstance(amount_a, (int, float))
            and isinstance(amount_b, (int, float))
            and not math.isclose(float(amount_a), float(amount_b), rel_tol=0.02, abs_tol=0.05)
        ):
            return "numeric_mismatch", 0.97
        return None
    if antonym_or_negation(left.get("value") or "", right.get("value") or ""):
        return "polarity_conflict", 0.94
    if opposite_polarity(left.get("polarity") or "neutral", right.get("polarity") or "neutral"):
        return "polarity_conflict", 0.95
    qty_a = parse_quantity(left.get("value") or "")
    qty_b = parse_quantity(right.get("value") or "")
    if qty_a and qty_b and qty_a[1] == qty_b[1] and not math.isclose(qty_a[0], qty_b[0], rel_tol=0.02, abs_tol=0.05):
        return "numeric_mismatch", 0.96
    amount_a = left.get("amount")
    amount_b = right.get("amount")
    if (
        isinstance(amount_a, (int, float))
        and isinstance(amount_b, (int, float))
        and not math.isclose(float(amount_a), float(amount_b), rel_tol=0.02, abs_tol=0.05)
    ):
        return "numeric_mismatch", 0.95
    return "value_mismatch", 0.8


def _iter_triples(graph: Any, brain_id: str, subject_uuids: list[str] | None = None):
    method = getattr(graph, "list_validity_relationships", None)
    skip = 0
    while skip < _MAX_SCAN:
        if callable(method):
            try:
                triples, total = method(
                    brain_id,
                    skip=skip,
                    limit=_REL_PAGE,
                    subject_uuids=subject_uuids,
                )
            except TypeError:
                triples, total = method(brain_id, skip=skip, limit=_REL_PAGE)
        else:
            result = graph.search_relationships(brain_id, limit=_REL_PAGE, skip=skip)
            items = getattr(result, "results", None) or result or []
            triples = [
                (item.subject, item.predicate, item.object) for item in items
            ]
            total = getattr(result, "total", None)
        if not triples:
            break
        for triple in triples:
            yield triple
        skip += len(triples)
        if total is not None and skip >= int(total):
            break


def _load_chunk_map(adapter: Any, brain_id: str, chunk_ids: list[str]) -> dict[str, Any]:
    unique = [cid for cid in dict.fromkeys(chunk_ids) if cid]
    if not unique or adapter is None:
        return {}
    try:
        result = adapter.get_text_chunks_by_ids(unique, False, brain_id)
    except Exception:
        return {}
    chunks = result[0] if isinstance(result, tuple) else result
    mapping: dict[str, Any] = {}
    for chunk in chunks or []:
        mapping[str(getattr(chunk, "id", "") or "")] = chunk
    return mapping


def _enrich_source(snapshot: dict[str, Any], chunks: dict[str, Any]) -> dict[str, Any]:
    enriched = dict(snapshot)
    for chunk_id in snapshot.get("source_chunk_ids") or []:
        chunk = chunks.get(chunk_id)
        if chunk is None:
            continue
        meta = getattr(chunk, "metadata", None) or {}
        if not isinstance(meta, dict):
            meta = {}
        if not enriched.get("source_key"):
            enriched["source_key"] = source_key_from_meta(meta)
        if not enriched.get("document"):
            enriched["document"] = (
                meta.get("title")
                or meta.get("url")
                or meta.get("document")
                or meta.get("source")
                or enriched.get("source_key")
            )
        text = (getattr(chunk, "text", None) or "").strip()
        if text:
            enriched["source_text"] = text[:500]
        break
    return enriched


def _side_from_snapshot(snapshot: dict[str, Any]) -> ConflictSide:
    return ConflictSide(
        fact_id=snapshot["fact_id"],
        subject=ConflictEntity(
            uuid=snapshot.get("subject_uuid") or "",
            name=snapshot.get("subject_name") or "",
            labels=list(snapshot.get("subject_labels") or []),
        ),
        predicate=snapshot.get("predicate") or "",
        object=ConflictEntity(
            uuid=snapshot.get("object_uuid") or "",
            name=snapshot.get("object_name") or "",
            labels=list(snapshot.get("object_labels") or []),
        ),
        value=snapshot.get("value") or "",
        polarity=snapshot.get("polarity") or "neutral",
        amount=snapshot.get("amount"),
        valid_from=snapshot.get(VALID_FROM),
        valid_to=snapshot.get(VALID_TO),
        superseded_by=snapshot.get(SUPERSEDED_BY),
        verification_state=snapshot.get("verification_state") or "unverified",
        source=ConflictSource(
            chunk_ids=list(snapshot.get("source_chunk_ids") or []),
            source_key=snapshot.get("source_key"),
            document=snapshot.get("document"),
            timestamp=snapshot.get("source_timestamp") or snapshot.get(VALID_FROM),
            text=snapshot.get("source_text"),
        ),
    )


def _build_conflict(
    left: dict[str, Any],
    right: dict[str, Any],
    disagreement_type: str,
    confidence: float,
    detection_method: str,
) -> FactConflict:
    pair = pair_key_for(left["fact_id"], right["fact_id"])
    return FactConflict(
        id=conflict_id_for(pair),
        pair_key=pair,
        claim_key=claim_key_for(left.get("subject_uuid") or "", left.get("predicate") or ""),
        status="open",
        disagreement_type=disagreement_type,  # type: ignore[arg-type]
        confidence=confidence,
        detection_method=detection_method,  # type: ignore[arg-type]
        detected_at=format_instant(utc_now()),
        subject=ConflictEntity(
            uuid=left.get("subject_uuid") or "",
            name=left.get("subject_name") or "",
            labels=list(left.get("subject_labels") or []),
        ),
        predicate=left.get("predicate") or "",
        sides=[_side_from_snapshot(left), _side_from_snapshot(right)],
        resolution=ConflictResolution(),
    )


def _configured_llm_client() -> Any | None:
    try:
        from src.core.instances import llm_small_adapter

        return llm_small_adapter
    except Exception:
        return None


def judge_semantic_pairs(
    pairs: list[tuple[dict[str, Any], dict[str, Any]]],
    *,
    client: Any | None = None,
    client_factory: Callable[[], Any | None] | None = None,
) -> dict[int, tuple[bool, float, str]]:
    """Return index -> (contradicts, confidence, type) for each pair.

    Uses the product's configured small LLM. Callers must bound ``pairs``.
    """
    if not pairs:
        return {}
    llm = client if client is not None else (client_factory or _configured_llm_client)()
    if llm is None:
        return {}
    payload = []
    for index, (left, right) in enumerate(pairs):
        payload.append(
            {
                "index": index,
                "subject": left.get("subject_name"),
                "predicate": left.get("predicate"),
                "a": {
                    "value": left.get("value"),
                    "polarity": left.get("polarity"),
                    "source": left.get("source_key"),
                },
                "b": {
                    "value": right.get("value"),
                    "polarity": right.get("polarity"),
                    "source": right.get("source_key"),
                },
            }
        )
    prompt = (
        "You are judging whether two knowledge-graph facts about the same "
        "subject and predicate contradict each other. Compatible restatements "
        "are not contradictions. Multi-valued facts (both can be true at once) "
        "are not contradictions. Reply with JSON only: "
        '{"results":[{"index":0,"contradicts":true,"confidence":0.0,'
        '"type":"semantic_contradiction"|"value_mismatch"|null}]}\n'
        f"Pairs: {payload}"
    )
    generate = getattr(llm, "generate_json", None)
    try:
        if callable(generate):
            parsed = generate(prompt, max_new_tokens=800)
        else:
            import json

            raw = llm.generate_text(prompt, max_new_tokens=800)
            text = str(raw or "").strip().strip("```json").strip("```").strip()
            parsed = json.loads(text)
    except Exception:
        return {}
    results: dict[int, tuple[bool, float, str]] = {}
    rows = parsed.get("results") if isinstance(parsed, dict) else parsed
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        try:
            index = int(row.get("index"))
        except (TypeError, ValueError):
            continue
        contradicts = bool(row.get("contradicts"))
        try:
            confidence = float(row.get("confidence") or 0.6)
        except (TypeError, ValueError):
            confidence = 0.6
        kind = str(row.get("type") or "semantic_contradiction")
        if kind not in {
            "value_mismatch",
            "numeric_mismatch",
            "polarity_conflict",
            "semantic_contradiction",
        }:
            kind = "semantic_contradiction"
        results[index] = (contradicts, max(0.0, min(confidence, 1.0)), kind)
    return results


def find_conflict_candidates(
    snapshots: list[dict[str, Any]],
) -> tuple[list[tuple[dict[str, Any], dict[str, Any], str, float]], list[tuple[dict[str, Any], dict[str, Any]]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for snapshot in snapshots:
        key = claim_key_for(snapshot.get("subject_uuid") or "", snapshot.get("predicate") or "")
        grouped.setdefault(key, []).append(snapshot)
    definite: list[tuple[dict[str, Any], dict[str, Any], str, float]] = []
    semantic: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for members in grouped.values():
        if len(members) < 2:
            continue
        predicate = members[0].get("predicate") or ""
        values = [item.get("value") or "" for item in members]
        if not is_functional_predicate(predicate, values):
            continue
        for i, left in enumerate(members):
            for right in members[i + 1 :]:
                if left.get("fact_id") == right.get("fact_id"):
                    continue
                if same_source_revision(left, right):
                    continue
                if not intervals_overlap(left, right):
                    continue
                classified = classify_disagreement(left, right)
                if classified is None:
                    continue
                kind, confidence = classified
                if kind == "value_mismatch" and not parse_quantity(left.get("value") or "") and not parse_quantity(right.get("value") or ""):
                    semantic.append((left, right))
                else:
                    definite.append((left, right, kind, confidence))
    return definite, semantic


def detect_conflicts(
    adapter: Any,
    graph: Any,
    brain_id: str,
    *,
    fact_ids: list[str] | None = None,
    subject_uuids: list[str] | None = None,
    use_llm: bool | None = None,
    llm_client: Any | None = None,
    llm_client_factory: Callable[[], Any | None] | None = None,
) -> list[FactConflict]:
    snapshots: list[dict[str, Any]] = []
    wanted = {str(item) for item in (fact_ids or []) if item}
    chunk_ids: list[str] = []
    subjects = list(subject_uuids or [])
    for triple in _iter_triples(graph, brain_id, subjects or None):
        snapshot = snapshot_from_triple(triple)
        if snapshot is None:
            continue
        snapshots.append(snapshot)
        chunk_ids.extend(snapshot.get("source_chunk_ids") or [])
    if wanted:
        keep_subjects = {
            item.get("subject_uuid")
            for item in snapshots
            if item.get("fact_id") in wanted
        }
        snapshots = [
            item
            for item in snapshots
            if item.get("fact_id") in wanted or item.get("subject_uuid") in keep_subjects
        ]
    chunks = _load_chunk_map(adapter, brain_id, chunk_ids)
    snapshots = [_enrich_source(item, chunks) for item in snapshots]
    definite, semantic = find_conflict_candidates(snapshots)
    found: dict[str, FactConflict] = {}
    for left, right, kind, confidence in definite:
        conflict = _build_conflict(left, right, kind, confidence, "deterministic")
        found[conflict.pair_key] = conflict
    llm_on = conflict_detect_llm_enabled() if use_llm is None else bool(use_llm)
    pending = [
        pair
        for pair in semantic
        if pair_key_for(pair[0]["fact_id"], pair[1]["fact_id"]) not in found
    ]
    if llm_on and pending:
        budget = conflict_detect_llm_max_pairs()
        judged: dict[int, tuple[bool, float, str]] = {}
        for start in range(0, min(len(pending), budget), _LLM_BATCH):
            batch = pending[start : start + _LLM_BATCH]
            batch_results = judge_semantic_pairs(
                batch, client=llm_client, client_factory=llm_client_factory
            )
            for index, verdict in batch_results.items():
                judged[start + index] = verdict
        for index, (left, right) in enumerate(pending):
            if index >= budget:
                break
            verdict = judged.get(index)
            if verdict is None:
                conflict = _build_conflict(left, right, "value_mismatch", 0.7, "deterministic")
                found[conflict.pair_key] = conflict
                continue
            contradicts, confidence, kind = verdict
            if not contradicts:
                continue
            conflict = _build_conflict(left, right, kind, confidence, "llm")
            found[conflict.pair_key] = conflict
    elif pending:
        for left, right in pending:
            conflict = _build_conflict(left, right, "value_mismatch", 0.7, "deterministic")
            found[conflict.pair_key] = conflict
    stored: list[FactConflict] = []
    for conflict in found.values():
        saved = upsert_conflict(adapter, brain_id, conflict)
        if saved is not None:
            stored.append(saved)
    return stored


def upsert_conflict(adapter: Any, brain_id: str, conflict: FactConflict) -> FactConflict | None:
    existing = None
    getter = getattr(adapter, "get_conflict_by_pair_key", None)
    if callable(getter):
        existing = getter(conflict.pair_key, brain_id)
    if existing is None:
        by_id = getattr(adapter, "get_conflict_by_id", None)
        if callable(by_id):
            existing = by_id(conflict.id, brain_id)
    current = conflict_from_document(existing) if isinstance(existing, dict) else existing
    if isinstance(current, FactConflict) and current.status in {"resolved", "dismissed"}:
        return current
    if isinstance(current, FactConflict):
        conflict.detected_at = current.detected_at
        conflict.status = current.status
        conflict.resolution = current.resolution
    saver = getattr(adapter, "save_conflict", None)
    if not callable(saver):
        return conflict
    saved = saver(conflict, brain_id)
    if isinstance(saved, FactConflict):
        return saved
    if isinstance(saved, dict):
        return conflict_from_document(saved)
    return conflict


def _optional_graph(graph: Any | None = None) -> Any | None:
    if graph is not None:
        return graph
    try:
        from src.core.instances import graph_adapter

        return graph_adapter
    except Exception:
        return None


def get_conflict(
    adapter: Any,
    brain_id: str,
    conflict_id: str,
    *,
    graph: Any | None = None,
) -> FactConflict | None:
    from src.core.saving.conflict_verification import hydrate_conflict_sides

    row = adapter.get_conflict_by_id(conflict_id, brain_id)
    if isinstance(row, FactConflict):
        conflict = row
    else:
        conflict = conflict_from_document(row if isinstance(row, dict) else None)
    return hydrate_conflict_sides(
        conflict, graph=_optional_graph(graph), brain_id=brain_id
    )


def list_conflicts(
    adapter: Any,
    brain_id: str,
    *,
    status: str | None = None,
    source: str | None = None,
    subject: str | None = None,
    since: Any = None,
    until: Any = None,
    limit: int = 10,
    skip: int = 0,
    graph: Any | None = None,
) -> tuple[list[FactConflict], int]:
    rows, total = adapter.get_conflicts_list(
        brain_id,
        limit=limit,
        skip=skip,
        status=status,
        source=source,
        subject=subject,
        since=since,
        until=until,
    )
    conflicts: list[FactConflict] = []
    for row in rows or []:
        if isinstance(row, FactConflict):
            conflicts.append(row)
        else:
            parsed = conflict_from_document(row if isinstance(row, dict) else None)
            if parsed is not None:
                conflicts.append(parsed)
    from src.core.saving.conflict_verification import hydrate_conflict_sides

    live_graph = _optional_graph(graph)
    hydrated = [
        hydrate_conflict_sides(item, graph=live_graph, brain_id=brain_id)
        for item in conflicts
    ]
    return hydrated, int(total or len(hydrated))


def _apply_resolution(
    adapter: Any,
    brain_id: str,
    conflict_id: str,
    *,
    status: str,
    method: str,
    winning_fact_id: str | None = None,
    notes: str | None = None,
    resolved_by: str = "api",
    graph: Any | None = None,
    apply_facts: bool = True,
) -> FactConflict:
    from src.core.saving.conflict_verification import (
        apply_verified_side_to_facts,
        hydrate_conflict_sides,
    )

    conflict = get_conflict(adapter, brain_id, conflict_id, graph=graph)
    if conflict is None:
        raise KeyError(conflict_id)
    live_graph = _optional_graph(graph)
    if winning_fact_id:
        side_ids = {side.fact_id for side in conflict.sides}
        if winning_fact_id not in side_ids:
            raise ValueError("winning_fact_id must be one of the conflict sides")
        if apply_facts and method == "verified_side" and status == "resolved":
            apply_verified_side_to_facts(
                live_graph,
                brain_id,
                winning_fact_id=winning_fact_id,
                losing_fact_ids=[
                    side.fact_id
                    for side in conflict.sides
                    if side.fact_id != winning_fact_id
                ],
                verified_by=resolved_by,
                note=notes,
            )
    conflict.status = status  # type: ignore[assignment]
    conflict.resolution = ConflictResolution(
        method=method,  # type: ignore[arg-type]
        winning_fact_id=winning_fact_id,
        notes=notes,
        resolved_at=format_instant(utc_now()),
        resolved_by=resolved_by,
    )
    hydrate_conflict_sides(conflict, graph=live_graph, brain_id=brain_id)
    return adapter.update_conflict(conflict, brain_id) or conflict


def resolve_conflict(
    adapter: Any,
    brain_id: str,
    conflict_id: str,
    *,
    winning_fact_id: str | None = None,
    notes: str | None = None,
    method: str = "manual",
    resolved_by: str = "api",
    graph: Any | None = None,
    apply_facts: bool = True,
) -> FactConflict:
    if method not in {"manual", "verified_side"}:
        method = "manual"
    return _apply_resolution(
        adapter,
        brain_id,
        conflict_id,
        status="resolved",
        method=method,
        winning_fact_id=winning_fact_id,
        notes=notes,
        resolved_by=resolved_by,
        graph=graph,
        apply_facts=apply_facts,
    )


def dismiss_conflict(
    adapter: Any,
    brain_id: str,
    conflict_id: str,
    *,
    notes: str | None = None,
    resolved_by: str = "api",
) -> FactConflict:
    return _apply_resolution(
        adapter,
        brain_id,
        conflict_id,
        status="dismissed",
        method="dismissed",
        notes=notes,
        resolved_by=resolved_by,
    )


def resolve_by_verified_fact(
    adapter: Any,
    brain_id: str,
    fact_id: str,
    *,
    notes: str | None = None,
    graph: Any | None = None,
) -> list[FactConflict]:
    """Close open conflicts after #113 verifies one side.

    Does not write fact states. #113's verify/reject endpoints should call
    this after they persist. Conflict resolve with method=verified_side is
    the inverse: it asks #113 to verify/reject, then closes the conflict.
    """
    rows = adapter.get_open_conflicts_for_facts([fact_id], brain_id)
    resolved: list[FactConflict] = []
    for row in rows or []:
        conflict = row if isinstance(row, FactConflict) else conflict_from_document(row)
        if conflict is None:
            continue
        resolved.append(
            resolve_conflict(
                adapter,
                brain_id,
                conflict.id,
                winning_fact_id=fact_id,
                notes=notes or "Resolved by verifying one side",
                method="verified_side",
                resolved_by="verification",
                graph=graph,
                apply_facts=False,
            )
        )
    return resolved


def open_conflicts_by_fact_ids(
    adapter: Any, brain_id: str, fact_ids: list[str]
) -> dict[str, list[FactConflict]]:
    unique = [item for item in dict.fromkeys(fact_ids) if item]
    if not unique:
        return {}
    rows = adapter.get_open_conflicts_for_facts(unique, brain_id)
    mapping: dict[str, list[FactConflict]] = {item: [] for item in unique}
    for row in rows or []:
        conflict = row if isinstance(row, FactConflict) else conflict_from_document(row)
        if conflict is None or conflict.status != "open":
            continue
        for side in conflict.sides:
            if side.fact_id in mapping:
                mapping[side.fact_id].append(conflict)
    return mapping


def annotation_payload(conflict: FactConflict, fact_id: str) -> dict[str, Any]:
    other = next((side for side in conflict.sides if side.fact_id != fact_id), None)
    return {
        "id": conflict.id,
        "disagreement_type": conflict.disagreement_type,
        "confidence": conflict.confidence,
        "predicate": conflict.predicate,
        "subject": conflict.subject.name,
        "other_value": other.value if other else None,
        "other_source": other.source.source_key if other else None,
    }


def maybe_detect_after_ingest(brain_id: str, fact_ids: list[str]) -> None:
    if not conflict_detect_on_ingest_enabled():
        return
    try:
        from src.core.saving.ingest_write_buffer import writes_are_staged

        if writes_are_staged():
            return
    except Exception:
        pass
    ids = [item for item in fact_ids if item]
    if not ids:
        return
    try:
        from src.workers.tasks.conflicts import detect_conflicts_incremental

        detect_conflicts_incremental.apply_async(
            args=[{"brain_id": brain_id, "fact_ids": ids}]
        )
        return
    except Exception:
        pass
    from src.services.data.main import data_adapter
    from src.services.kg_agent.main import graph_adapter

    detect_conflicts(data_adapter, graph_adapter, brain_id, fact_ids=ids)
