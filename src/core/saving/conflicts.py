"""Detect and store cross-source conflicts between graph facts.

Deterministic matching is the default. An optional batched call through the
configured small LLM judges leftover semantic pairs. Ingest-time detection
is gated by ``CONFLICT_DETECT_ON_INGEST`` (default off).
"""

from __future__ import annotations

import hashlib
import inspect
import logging
import math
import os
import re
from typing import Any, Callable, Iterable

logger = logging.getLogger(__name__)

from src.constants.conflicts import (
    ConflictEntity,
    ConflictResolution,
    ConflictSide,
    ConflictSource,
    FactConflict,
    conflict_from_document,
    conflict_to_document,
)
from src.core.auth.scopes import VERIFICATION_PERMISSION_DENIED
from src.core.saving.source_passages import source_key_from_meta
from src.core.saving.validity import (
    SUPERSEDED_BY,
    VALID_FROM,
    VALID_TO,
    format_instant,
    parse_instant,
    utc_now,
)

logger = logging.getLogger(__name__)

_REL_PAGE = 200
_LLM_BATCH = 10


class VerificationPermissionDenied(PermissionError):
    code = VERIFICATION_PERMISSION_DENIED


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


def conflict_detect_max_relationships() -> int:
    """Relationship cap for a conflict scan.

    Default is 10000. ``0`` means unlimited (load every relationship).
    Unset env falls through to ``config.conflict_detect_max_relationships``.
    """
    raw = os.getenv("CONFLICT_DETECT_MAX_RELATIONSHIPS")
    if raw is not None and str(raw).strip() != "":
        try:
            return max(0, int(raw))
        except ValueError:
            return 10_000
    try:
        from src.config import config

        return max(
            0, int(getattr(config, "conflict_detect_max_relationships", 10_000) or 0)
        )
    except Exception:
        return 10_000


def _effective_max_relationships(max_relationships: int | None) -> int | None:
    cap = (
        conflict_detect_max_relationships()
        if max_relationships is None
        else int(max_relationships)
    )
    if cap <= 0:
        return None
    return cap


class ConflictScanResult:
    """Conflicts plus scan completeness metadata.

    Iterates like the stored conflict list so existing callers keep working.
    """

    def __init__(
        self,
        conflicts: list[FactConflict] | None = None,
        *,
        truncated: bool = False,
        scanned_relationships: int = 0,
        total_relationships: int | None = None,
        llm_truncated: bool = False,
        llm_pairs_total: int = 0,
        llm_pairs_attempted: int = 0,
        llm_pairs_judged: int = 0,
        llm_pairs_failed: int = 0,
        stale_resolved: int = 0,
        scan_cursor: dict[str, str] | None = None,
        scanned_claim_keys: list[str] | None = None,
    ):
        self.conflicts = list(conflicts or [])
        self.truncated = bool(truncated)
        self.scanned_relationships = int(scanned_relationships)
        self.total_relationships = (
            None if total_relationships is None else int(total_relationships)
        )
        self.llm_truncated = bool(llm_truncated)
        self.llm_pairs_total = int(llm_pairs_total)
        self.llm_pairs_attempted = int(llm_pairs_attempted)
        self.llm_pairs_judged = int(llm_pairs_judged)
        self.llm_pairs_failed = int(llm_pairs_failed)
        self.stale_resolved = int(stale_resolved)
        self.scan_cursor = dict(scan_cursor) if scan_cursor else None
        self.scanned_claim_keys = list(scanned_claim_keys or [])

    def __iter__(self):
        return iter(self.conflicts)

    def __len__(self):
        return len(self.conflicts)

    def __getitem__(self, index):
        return self.conflicts[index]

    def __bool__(self):
        return bool(self.conflicts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "count": len(self.conflicts),
            "conflict_ids": [item.id for item in self.conflicts],
            "truncated": self.truncated,
            "scanned_relationships": self.scanned_relationships,
            "total_relationships": self.total_relationships,
            "llm_truncated": self.llm_truncated,
            "llm_pairs_total": self.llm_pairs_total,
            "llm_pairs_attempted": self.llm_pairs_attempted,
            "llm_pairs_judged": self.llm_pairs_judged,
            "llm_pairs_failed": self.llm_pairs_failed,
            "stale_resolved": self.stale_resolved,
            "scan_cursor": self.scan_cursor,
        }


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
    if parse_date_value(value_text) is not None:
        return None
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


def parse_date_value(text: str | None):
    if not text or not str(text).strip():
        return None
    from src.core.saving.validity import parse_instant
    from src.utils.dates import parse_date_string

    parsed = parse_instant(text)
    if parsed is not None:
        return parsed
    return parse_date_string(str(text).strip())


def _date_key(value) -> tuple[int, int, int] | None:
    if value is None:
        return None
    naive = value.replace(tzinfo=None) if getattr(value, "tzinfo", None) else value
    try:
        return (naive.year, naive.month, naive.day)
    except Exception:
        return None


_TIME_RE = re.compile(
    r"T(?:\d{1,2}:\d{2}(?::\d{2}(?:[.,]\d+)?)?|\d{4,6}(?:[.,]\d+)?)"
    r"|\d{1,2}:\d{2}(?::\d{2})?(?:\s*[APap][Mm])?"
)


def _has_clock_time(text: str | None) -> bool:
    return bool(_TIME_RE.search(str(text or "")))


def _instant_key(value) -> tuple | None:
    if value is None:
        return None
    naive = value.replace(tzinfo=None) if getattr(value, "tzinfo", None) else value
    try:
        return (
            naive.year,
            naive.month,
            naive.day,
            naive.hour,
            naive.minute,
            naive.second,
        )
    except Exception:
        return None


def values_equivalent(left: str, right: str) -> bool:
    a = normalize_value(left)
    b = normalize_value(right)
    if a == b:
        return True
    date_a = parse_date_value(left)
    date_b = parse_date_value(right)
    if date_a is not None and date_b is not None:
        if _has_clock_time(left) and _has_clock_time(right):
            return _instant_key(date_a) == _instant_key(date_b)
        return _date_key(date_a) == _date_key(date_b)
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


def _structurally_incompatible(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_value = left.get("value") or ""
    right_value = right.get("value") or ""
    if parse_quantity(left_value) and parse_quantity(right_value):
        return True
    if parse_date_value(left_value) and parse_date_value(right_value):
        return True
    if antonym_or_negation(left_value, right_value):
        return True
    amount_a = left.get("amount")
    amount_b = right.get("amount")
    return isinstance(amount_a, (int, float)) and isinstance(amount_b, (int, float))


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
    left_value = left.get("value") or ""
    right_value = right.get("value") or ""
    date_a = parse_date_value(left_value)
    date_b = parse_date_value(right_value)
    if date_a is not None and date_b is not None:
        if _has_clock_time(left_value) and _has_clock_time(right_value):
            if _instant_key(date_a) != _instant_key(date_b):
                return "value_mismatch", 0.96
        elif _date_key(date_a) != _date_key(date_b):
            return "value_mismatch", 0.96
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


def _triple_subject_uuid(triple: Any) -> str:
    if not triple:
        return ""
    return str(getattr(triple[0], "uuid", "") or "")


def _triple_relationship_uuid(triple: Any) -> str:
    if not triple or len(triple) < 2:
        return ""
    predicate = triple[1]
    props = _props(predicate)
    return str(getattr(predicate, "uuid", None) or props.get("uuid") or "")


def _triple_predicate_name(triple: Any) -> str:
    if not triple or len(triple) < 2:
        return ""
    predicate = triple[1]
    props = _props(predicate)
    return str(getattr(predicate, "name", "") or props.get("rel_type") or "")


def _group_key_for(subject_uuid: str | None, predicate: str | None) -> tuple[str, str]:
    return (str(subject_uuid or ""), normalize_predicate(predicate or ""))


def _triple_group_key(triple: Any) -> tuple[str, str]:
    return _group_key_for(_triple_subject_uuid(triple), _triple_predicate_name(triple))


def _relationship_sort_key(
    subject_uuid: str | None,
    predicate: str | None,
    relationship_uuid: str | None,
) -> tuple[str, str, str]:
    return (
        str(subject_uuid or ""),
        normalize_predicate(predicate or ""),
        str(relationship_uuid or ""),
    )


def _triple_sort_key(triple: Any) -> tuple[str, str, str]:
    return _relationship_sort_key(
        _triple_subject_uuid(triple),
        _triple_predicate_name(triple),
        _triple_relationship_uuid(triple),
    )


def _cursor_group(
    after_subject_uuid: str | None, after_predicate: str | None
) -> tuple[str, str] | None:
    subject = str(after_subject_uuid or "").strip()
    predicate = str(after_predicate or "").strip()
    if not subject and not predicate:
        return None
    return _group_key_for(subject, predicate)


def _sort_validity_triples(triples: Iterable[Any]) -> list[Any]:
    return sorted(list(triples or []), key=_triple_sort_key)


def _triples_after_cursor(
    triples: Iterable[Any],
    *,
    after_subject_uuid: str | None = None,
    after_predicate: str | None = None,
    after_relationship_uuid: str | None = None,
) -> list[Any]:
    items = list(triples or [])
    rel = str(after_relationship_uuid or "").strip()
    if rel:
        cursor = (
            str(after_subject_uuid or ""),
            normalize_predicate(after_predicate or ""),
            rel,
        )
        return [item for item in items if _triple_sort_key(item) > cursor]
    group = _cursor_group(after_subject_uuid, after_predicate)
    if group is None:
        return items
    return [item for item in items if _triple_group_key(item) > group]


def _search_relationship_rows(result: Any) -> list[Any]:
    if result is None:
        return []
    if hasattr(result, "results"):
        return list(getattr(result, "results") or [])
    if isinstance(result, (list, tuple)):
        return list(result)
    return []


def _collect_search_relationship_triples(
    graph: Any,
    brain_id: str,
    *,
    search: Callable[..., Any] | None = None,
    page: int | None = None,
    stop_relationship_uuids: Iterable[str] | None = None,
) -> tuple[list[Any], int]:
    method = search or getattr(graph, "search_relationships", None)
    if not callable(method):
        return [], 0
    collected: list[Any] = []
    cursor = 0
    batch_size = max(int(page or _REL_PAGE), _REL_PAGE)
    reported_total = None
    wanted = {str(item) for item in (stop_relationship_uuids or []) if item}
    found: set[str] = set()
    while True:
        result = method(brain_id, batch_size, cursor)
        rows = _search_relationship_rows(result)
        total = getattr(result, "total", None) if result is not None else None
        if total is not None:
            reported_total = int(total)
        batch = [
            (item.subject, item.predicate, item.object) for item in rows
        ]
        if not batch:
            break
        collected.extend(batch)
        cursor += len(batch)
        if wanted:
            found.update(
                _triple_relationship_uuid(item)
                for item in batch
                if _triple_relationship_uuid(item) in wanted
            )
            if wanted <= found:
                break
        if reported_total is not None and cursor >= reported_total:
            break
        if len(batch) < batch_size:
            break
    return collected, (
        reported_total if reported_total is not None else len(collected)
    )


def _read_scan_cursor(adapter: Any, brain_id: str) -> dict[str, str] | None:
    getter = getattr(adapter, "get_conflict_scan_cursor", None)
    if not callable(getter):
        return None
    try:
        row = getter(brain_id)
    except TypeError:
        try:
            row = getter(brain_id=brain_id)
        except TypeError:
            return None
        except Exception:
            logger.exception(
                "conflict scan cursor read failed brain_id=%s", brain_id
            )
            raise
    except Exception:
        logger.exception("conflict scan cursor read failed brain_id=%s", brain_id)
        raise
    if not isinstance(row, dict):
        return None
    subject = str(row.get("after_subject") or "").strip()
    predicate = str(row.get("after_predicate") or "").strip()
    if not subject and not predicate:
        return None
    return {"after_subject": subject, "after_predicate": predicate}


def _write_scan_cursor(
    adapter: Any, brain_id: str, cursor: dict[str, str] | None
) -> None:
    setter = getattr(adapter, "set_conflict_scan_cursor", None)
    if not callable(setter):
        return
    payload = None
    if cursor:
        subject = str(cursor.get("after_subject") or "").strip()
        predicate = str(cursor.get("after_predicate") or "").strip()
        if subject or predicate:
            payload = {"after_subject": subject, "after_predicate": predicate}
    try:
        setter(brain_id, payload)
        return
    except TypeError:
        pass
    except Exception:
        logger.exception(
            "conflict scan cursor write failed brain_id=%s", brain_id
        )
        raise
    try:
        setter(brain_id=brain_id, cursor=payload)
    except TypeError:
        return
    except Exception:
        logger.exception(
            "conflict scan cursor write failed brain_id=%s", brain_id
        )
        raise


def _filter_validity_triples(
    triples: Iterable[Any],
    subject_uuids: list[str] | None = None,
    relationship_uuids: list[str] | None = None,
) -> list[Any]:
    items = list(triples or [])
    if subject_uuids:
        allowed = set(subject_uuids)
        items = [item for item in items if _triple_subject_uuid(item) in allowed]
    if relationship_uuids:
        allowed_rels = set(relationship_uuids)
        items = [
            item
            for item in items
            if _triple_relationship_uuid(item) in allowed_rels
        ]
    return items


def _bind_supported_kwargs(method: Callable[..., Any], kwargs: dict[str, Any]) -> dict[str, Any]:
    try:
        signature = inspect.signature(method)
    except (TypeError, ValueError):
        return dict(kwargs)
    if any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    ):
        return dict(kwargs)
    return {
        key: value
        for key, value in kwargs.items()
        if key in signature.parameters
    }


def _call_list_validity_relationships(
    method: Callable[..., Any],
    brain_id: str,
    *,
    skip: int,
    limit: int,
    subject_uuids: list[str] | None = None,
    relationship_uuids: list[str] | None = None,
    after_subject_uuid: str | None = None,
    after_predicate: str | None = None,
    after_relationship_uuid: str | None = None,
):
    requested = {"skip": skip, "limit": limit}
    if subject_uuids:
        requested["subject_uuids"] = subject_uuids
    if relationship_uuids:
        requested["relationship_uuids"] = relationship_uuids
    if after_subject_uuid:
        requested["after_subject_uuid"] = after_subject_uuid
    if after_predicate:
        requested["after_predicate"] = after_predicate
    if after_relationship_uuid:
        requested["after_relationship_uuid"] = after_relationship_uuid
    accepted = _bind_supported_kwargs(method, requested)
    missing_filters = [
        key
        for key in ("subject_uuids", "relationship_uuids")
        if key in requested and key not in accepted
    ]
    if not missing_filters:
        triples, total = method(brain_id, **accepted)
        if after_subject_uuid or after_predicate or after_relationship_uuid:
            if "after_subject_uuid" not in accepted and "after_predicate" not in accepted:
                triples = _triples_after_cursor(
                    _sort_validity_triples(triples),
                    after_subject_uuid=after_subject_uuid,
                    after_predicate=after_predicate,
                    after_relationship_uuid=after_relationship_uuid,
                )
        return triples, total
    collected: list[Any] = []
    cursor = 0
    page = max(int(limit or _REL_PAGE), _REL_PAGE)
    reported_total = None
    while True:
        page_kwargs = dict(accepted)
        page_kwargs["skip"] = cursor
        page_kwargs["limit"] = page
        triples, total = method(brain_id, **page_kwargs)
        if total is not None:
            reported_total = int(total)
        batch = list(triples or [])
        if not batch:
            break
        collected.extend(batch)
        cursor += len(batch)
        if reported_total is not None and cursor >= reported_total:
            break
        if len(batch) < page:
            break
    filtered = _filter_validity_triples(
        collected,
        subject_uuids=subject_uuids,
        relationship_uuids=relationship_uuids,
    )
    ordered = _sort_validity_triples(filtered)
    ordered = _triples_after_cursor(
        ordered,
        after_subject_uuid=after_subject_uuid,
        after_predicate=after_predicate,
        after_relationship_uuid=after_relationship_uuid,
    )
    return ordered[skip : skip + limit], len(filtered)


def _load_triples(
    graph: Any,
    brain_id: str,
    subject_uuids: list[str] | None = None,
    relationship_uuids: list[str] | None = None,
    *,
    max_relationships: int | None = None,
    after_subject_uuid: str | None = None,
    after_predicate: str | None = None,
    rotate: bool = False,
) -> tuple[list[Any], dict[str, Any]]:
    method = getattr(graph, "list_validity_relationships", None)
    start_cursor = _cursor_group(after_subject_uuid, after_predicate)
    after_s = after_subject_uuid or None
    after_p = after_predicate or None
    after_r: str | None = None
    skip = 0
    scanned: list[Any] = []
    scanned_groups: list[tuple[str, str]] = []
    seen_groups: set[tuple[str, str]] = set()
    last_group: tuple[str, str] | None = None
    last_predicate_raw = ""
    reported_total = None
    truncated = False
    wrapped = False
    pending: list[Any] = []
    pending_key: tuple[str, str] | None = None
    pending_raw = ""
    keyset = True
    stopped_for_cap = False
    search_ordered: tuple[list[Any], Any] | None = None

    def fetch(page_limit: int):
        nonlocal keyset, search_ordered
        if callable(method):
            requested = {
                "skip": 0 if keyset else skip,
                "limit": page_limit,
            }
            if after_s:
                requested["after_subject_uuid"] = after_s
            if after_p:
                requested["after_predicate"] = after_p
            if after_r:
                requested["after_relationship_uuid"] = after_r
            accepted = _bind_supported_kwargs(method, requested)
            if after_s or after_p or after_r:
                if (
                    "after_subject_uuid" not in accepted
                    and "after_predicate" not in accepted
                    and "after_relationship_uuid" not in accepted
                ):
                    keyset = False
            return _call_list_validity_relationships(
                method,
                brain_id,
                skip=0 if keyset else skip,
                limit=page_limit,
                subject_uuids=subject_uuids,
                relationship_uuids=relationship_uuids,
                after_subject_uuid=after_s if keyset else None,
                after_predicate=after_p if keyset else None,
                after_relationship_uuid=after_r if keyset else None,
            )
        keyset = False
        if search_ordered is None:
            collected, search_total = _collect_search_relationship_triples(
                graph, brain_id
            )
            search_ordered = (
                _sort_validity_triples(collected),
                search_total,
            )
        triples, total = search_ordered
        sliced = _triples_after_cursor(
            triples,
            after_subject_uuid=after_s,
            after_predicate=after_p,
            after_relationship_uuid=after_r,
        )
        return sliced[:page_limit], total

    def flush_pending() -> bool:
        nonlocal pending, pending_key, pending_raw, last_group, last_predicate_raw, truncated
        if not pending or pending_key is None:
            pending = []
            pending_key = None
            pending_raw = ""
            return True
        if pending_key in seen_groups:
            pending = []
            pending_key = None
            pending_raw = ""
            return True
        size = len(pending)
        if (
            max_relationships is not None
            and size > max_relationships
            and not scanned
        ):
            logger.warning(
                "conflict scan including oversized group brain_id=%s size=%s cap=%s",
                brain_id,
                size,
                max_relationships,
            )
        if (
            max_relationships is not None
            and scanned
            and len(scanned) + size > max_relationships
        ):
            truncated = True
            return False
        scanned.extend(pending)
        scanned_groups.append(pending_key)
        seen_groups.add(pending_key)
        last_group = pending_key
        last_predicate_raw = pending_raw
        pending = []
        pending_key = None
        pending_raw = ""
        return True

    def add_triple(triple: Any) -> bool:
        nonlocal pending, pending_key, pending_raw
        key = _triple_group_key(triple)
        if not key[0]:
            return True
        if not wrapped and start_cursor is not None and key <= start_cursor:
            return True
        if wrapped and start_cursor is not None and key > start_cursor:
            return flush_pending() and False
        if key in seen_groups:
            return True
        if pending_key is None:
            pending_key = key
            pending = [triple]
            pending_raw = _triple_predicate_name(triple)
            return True
        if key == pending_key:
            pending.append(triple)
            return True
        if not flush_pending():
            return False
        pending_key = key
        pending = [triple]
        pending_raw = _triple_predicate_name(triple)
        return True

    while True:
        page_limit = _REL_PAGE
        triples, total = fetch(page_limit)
        if total is not None:
            reported_total = int(total)
        batch = list(triples or [])
        if not batch:
            if rotate and not wrapped and start_cursor is not None:
                wrapped = True
                flush_pending()
                after_s = after_p = after_r = None
                skip = 0
                continue
            flush_pending()
            break
        stopped = False
        for triple in batch:
            if not add_triple(triple):
                stopped = True
                stopped_for_cap = True
                break
        if stopped:
            break
        last = batch[-1]
        after_s, after_p = _triple_group_key(last)
        after_p = _triple_predicate_name(last)
        after_s = _triple_subject_uuid(last)
        after_r = _triple_relationship_uuid(last)
        skip += len(batch)
        if len(batch) < page_limit:
            if rotate and not wrapped and start_cursor is not None:
                wrapped = True
                flush_pending()
                after_s = after_p = after_r = None
                skip = 0
                continue
            flush_pending()
            break
        if (
            max_relationships is not None
            and scanned
            and len(scanned) >= max_relationships
        ):
            truncated = reported_total is None or reported_total > len(scanned)
            stopped_for_cap = True
            break

    if pending and not stopped_for_cap:
        flush_pending()

    if max_relationships is not None:
        if reported_total is not None and reported_total > len(scanned):
            truncated = True
        elif reported_total is None and len(scanned) >= max_relationships:
            truncated = True
    if reported_total is not None and len(scanned) >= reported_total:
        truncated = False

    cursor_out = None
    if last_group is not None:
        cursor_out = {
            "after_subject": last_group[0],
            "after_predicate": last_predicate_raw or last_group[1],
        }
    stats = {
        "scanned": len(scanned),
        "total": reported_total,
        "truncated": bool(truncated),
        "groups": scanned_groups,
        "scan_cursor": cursor_out,
    }
    return scanned, stats


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
        subject_uuid = str(snapshot.get("subject_uuid") or "").strip()
        predicate = str(snapshot.get("predicate") or "").strip()
        if not subject_uuid or not predicate:
            continue
        key = claim_key_for(subject_uuid, predicate)
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
                if (left.get("subject_uuid") or "") != (right.get("subject_uuid") or ""):
                    continue
                if normalize_predicate(left.get("predicate") or "") != normalize_predicate(
                    right.get("predicate") or ""
                ):
                    continue
                if left.get("deprecated") or right.get("deprecated"):
                    continue
                if same_source_revision(left, right):
                    continue
                if not intervals_overlap(left, right):
                    continue
                classified = classify_disagreement(left, right)
                if classified is None:
                    continue
                kind, confidence = classified
                if kind == "value_mismatch" and not _structurally_incompatible(left, right):
                    semantic.append((left, right))
                else:
                    definite.append((left, right, kind, confidence))
    return definite, semantic


def _subjects_for_fact_ids(graph: Any, brain_id: str, fact_ids: list[str]) -> list[str]:
    wanted = [item for item in dict.fromkeys(fact_ids) if item]
    if not wanted:
        return []
    triples, _stats = _load_triples(
        graph, brain_id, relationship_uuids=wanted, max_relationships=len(wanted)
    )
    subjects: list[str] = []
    seen: set[str] = set()
    for triple in triples:
        snapshot = snapshot_from_triple(triple)
        if snapshot is None or snapshot.get("fact_id") not in set(wanted):
            continue
        subject = str(snapshot.get("subject_uuid") or "").strip()
        if subject and subject not in seen:
            seen.add(subject)
            subjects.append(subject)
    return subjects


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
    max_relationships: int | None = None,
) -> ConflictScanResult:
    wanted = {str(item) for item in (fact_ids or []) if item}
    subjects = [item for item in dict.fromkeys(subject_uuids or []) if item]
    if wanted and not subjects:
        subjects = _subjects_for_fact_ids(graph, brain_id, list(wanted))
    scoped = bool(subjects or wanted)
    cursor = None if scoped else _read_scan_cursor(adapter, brain_id)
    triples, scan_stats = _load_triples(
        graph,
        brain_id,
        subjects or None,
        max_relationships=_effective_max_relationships(max_relationships),
        after_subject_uuid=(cursor or {}).get("after_subject"),
        after_predicate=(cursor or {}).get("after_predicate"),
        rotate=not scoped,
    )
    snapshots: list[dict[str, Any]] = []
    chunk_ids: list[str] = []
    for triple in triples:
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
    llm_pairs_total = len(pending)
    llm_pairs_attempted = 0
    llm_pairs_judged = 0
    llm_pairs_failed = 0
    llm_truncated = False
    judged_pair_keys: set[str] = set()
    compatible_pair_keys: set[str] = set()
    if llm_on and pending:
        budget = conflict_detect_llm_max_pairs()
        limit = len(pending) if budget <= 0 else min(len(pending), budget)
        llm_truncated = budget > 0 and len(pending) > budget
        llm_pairs_attempted = limit
        judged: dict[int, tuple[bool, float, str]] = {}
        for start in range(0, limit, _LLM_BATCH):
            batch = pending[start : start + _LLM_BATCH]
            batch_results = judge_semantic_pairs(
                batch, client=llm_client, client_factory=llm_client_factory
            )
            for index, verdict in batch_results.items():
                judged[start + index] = verdict
        for index, (left, right) in enumerate(pending[:limit]):
            pair = pair_key_for(left["fact_id"], right["fact_id"])
            verdict = judged.get(index)
            if verdict is None:
                continue
            judged_pair_keys.add(pair)
            contradicts, confidence, kind = verdict
            if not contradicts:
                compatible_pair_keys.add(pair)
                continue
            conflict = _build_conflict(left, right, kind, confidence, "llm")
            found[conflict.pair_key] = conflict
        llm_pairs_judged = len(judged_pair_keys)
        llm_pairs_failed = max(0, llm_pairs_attempted - llm_pairs_judged)
        if llm_truncated:
            logger.warning(
                "conflict LLM judging truncated brain_id=%s judged=%s attempted=%s total=%s",
                brain_id,
                llm_pairs_judged,
                llm_pairs_attempted,
                llm_pairs_total,
            )
    stored: list[FactConflict] = []
    for conflict in found.values():
        saved = upsert_conflict(adapter, brain_id, conflict)
        if saved is not None:
            stored.append(saved)
    scanned_claim_keys = {
        claim_key_for(item.get("subject_uuid") or "", item.get("predicate") or "")
        for item in snapshots
        if item.get("subject_uuid") and item.get("predicate")
    }
    truncated = bool(scan_stats.get("truncated"))
    stale: list[FactConflict] = []
    stale = _stale_open_conflicts(
        adapter,
        brain_id,
        snapshots,
        live_pair_keys=set(found),
        fact_ids=list(wanted) or None,
        scoped=scoped,
        graph=graph,
        llm_on=llm_on,
        llm_truncated=llm_truncated,
        judged_pair_keys=judged_pair_keys,
        compatible_pair_keys=compatible_pair_keys,
        scanned_claim_keys=scanned_claim_keys if truncated else None,
    )
    stored.extend(stale)
    if truncated:
        logger.warning(
            "conflict relationship scan truncated brain_id=%s scanned=%s total=%s",
            brain_id,
            scan_stats.get("scanned"),
            scan_stats.get("total"),
        )
    if not scoped:
        next_cursor = None if not truncated else scan_stats.get("scan_cursor")
        _write_scan_cursor(adapter, brain_id, next_cursor)
    return ConflictScanResult(
        stored,
        truncated=truncated,
        scanned_relationships=int(scan_stats.get("scanned") or 0),
        total_relationships=scan_stats.get("total"),
        llm_truncated=llm_truncated,
        llm_pairs_total=llm_pairs_total,
        llm_pairs_attempted=llm_pairs_attempted,
        llm_pairs_judged=llm_pairs_judged,
        llm_pairs_failed=llm_pairs_failed,
        stale_resolved=len(stale),
        scan_cursor=None if not truncated else scan_stats.get("scan_cursor"),
        scanned_claim_keys=sorted(scanned_claim_keys),
    )


def _conflict_from_row(row: Any) -> FactConflict | None:
    if isinstance(row, FactConflict):
        return row
    if isinstance(row, dict):
        return conflict_from_document(row)
    return None


def _iter_open_conflicts(
    adapter: Any,
    brain_id: str,
    fact_ids: list[str] | None = None,
):
    collected: list[FactConflict] = []
    if fact_ids is not None:
        if not fact_ids:
            return
        getter = getattr(adapter, "get_open_conflicts_for_facts", None)
        if callable(getter):
            for row in getter(fact_ids, brain_id) or []:
                conflict = _conflict_from_row(row)
                if conflict is not None and conflict.status == "open":
                    collected.append(conflict)
            for conflict in collected:
                yield conflict
            return
    listing = getattr(adapter, "get_conflicts_list", None)
    if not callable(listing):
        return
    skip = 0
    page = 200
    while True:
        rows, total = listing(brain_id, limit=page, skip=skip, status="open")
        batch = list(rows or [])
        if not batch:
            break
        for row in batch:
            conflict = _conflict_from_row(row)
            if conflict is not None and conflict.status == "open":
                collected.append(conflict)
        skip += len(batch)
        if total is not None and skip >= int(total):
            break
        if len(batch) < page:
            break
    if fact_ids is not None:
        wanted = set(fact_ids)
        collected = [
            conflict
            for conflict in collected
            if wanted.intersection(conflict.fact_ids())
        ]
    for conflict in collected:
        yield conflict


def _stale_reason(
    conflict: FactConflict,
    snapshots_by_id: dict[str, dict[str, Any]],
    live_pair_keys: set[str],
    compatible_pair_keys: set[str] | None = None,
) -> str | None:
    """Return a concrete stale reason, or None.

    Allowed reasons: a side was removed/deprecated, a side was superseded,
    validity windows no longer overlap, or the values are now compatible.
    Remaining disagreement is not stale; callers must not invent a catch-all.
    """
    sides = list(conflict.sides or [])
    if len(sides) < 2:
        return "side_removed:incomplete_sides"
    live = [snapshots_by_id.get(side.fact_id) for side in sides]
    missing = [side.fact_id for side, snap in zip(sides, live) if snap is None]
    if missing:
        return "side_removed:" + ",".join(missing)
    deprecated = [
        snap["fact_id"] for snap in live if snap is not None and snap.get("deprecated")
    ]
    if deprecated:
        return "side_removed:" + ",".join(deprecated)
    pair = pair_key_for(sides[0].fact_id, sides[1].fact_id)
    if pair in live_pair_keys:
        return None
    left, right = live[0], live[1]
    if left is None or right is None:
        return "side_removed"
    successor_ids = {
        str(snap.get(SUPERSEDED_BY) or "")
        for snap in live
        if snap.get(SUPERSEDED_BY)
    }
    successor_ids.discard("")
    if successor_ids:
        other = right if left.get(SUPERSEDED_BY) else left
        successors = [snapshots_by_id.get(item) for item in successor_ids]
        if any(
            snap is not None
            and other is not None
            and classify_disagreement(snap, other) is None
            for snap in successors
        ):
            return "side_superseded:" + ",".join(sorted(successor_ids))
    if same_source_revision(left, right):
        return "side_superseded:same_source_revision"
    if not intervals_overlap(left, right):
        return "no_longer_holds:validity_no_overlap"
    if classify_disagreement(left, right) is None:
        return "no_longer_holds:values_compatible"
    if compatible_pair_keys and pair in compatible_pair_keys:
        return "no_longer_holds:llm_compatible"
    return None


def _skip_llm_stale(
    conflict: FactConflict,
    reason: str,
    *,
    llm_on: bool,
    llm_truncated: bool,
    judged_pair_keys: set[str],
) -> bool:
    """LLM conflicts stay open unless this scan could re-judge them.

    Concrete graph evidence (removed/deprecated/superseded sides, or
    non-overlapping validity) still closes. Heuristic ``values_compatible``
    does not close an LLM conflict when the LLM was off, the pair budget
    truncated the scan, or this pair got no verdict.
    """
    if (conflict.detection_method or "") != "llm":
        return False
    if reason.startswith("side_removed") or reason.startswith("side_superseded"):
        return False
    if reason.startswith("no_longer_holds:validity_no_overlap"):
        return False
    pair = pair_key_for(*conflict.fact_ids()[:2]) if len(conflict.fact_ids()) >= 2 else ""
    if pair and pair in judged_pair_keys:
        return False
    if not llm_on or llm_truncated:
        return True
    return True


def _stale_open_conflicts(
    adapter: Any,
    brain_id: str,
    snapshots: list[dict[str, Any]],
    *,
    live_pair_keys: set[str],
    fact_ids: list[str] | None = None,
    scoped: bool = False,
    graph: Any | None = None,
    llm_on: bool = False,
    llm_truncated: bool = False,
    judged_pair_keys: set[str] | None = None,
    compatible_pair_keys: set[str] | None = None,
    scanned_claim_keys: set[str] | None = None,
) -> list[FactConflict]:
    snapshots_by_id = {
        str(item.get("fact_id")): item
        for item in snapshots
        if item.get("fact_id")
    }
    lookup_ids = list(
        dict.fromkeys([*(fact_ids or []), *snapshots_by_id.keys()])
    )
    claim_scoped = scanned_claim_keys is not None
    if scoped and not lookup_ids and not claim_scoped:
        return []
    window_candidates: list[FactConflict] = []
    removed_group_candidates: list[FactConflict] = []
    seen: set[str] = set()
    refresh_ids: list[str] = []
    for conflict in _iter_open_conflicts(
        adapter, brain_id, lookup_ids if scoped and not claim_scoped else None
    ):
        if conflict.id in seen or conflict.pair_key in live_pair_keys:
            continue
        seen.add(conflict.id)
        side_ids = [
            side.fact_id for side in conflict.sides if side.fact_id
        ]
        in_window = (not claim_scoped) or (
            conflict.claim_key in scanned_claim_keys
        )
        if in_window:
            window_candidates.append(conflict)
            if any(item not in snapshots_by_id for item in side_ids):
                refresh_ids.extend(side_ids)
        elif graph is not None:
            removed_group_candidates.append(conflict)
            refresh_ids.extend(side_ids)
    wanted_unknown = [item for item in dict.fromkeys(refresh_ids) if item]
    if wanted_unknown and graph is not None:
        triples, _stats = _load_triples(
            graph,
            brain_id,
            relationship_uuids=wanted_unknown,
            max_relationships=len(wanted_unknown),
        )
        for triple in triples:
            snapshot = snapshot_from_triple(triple)
            fact_id = str((snapshot or {}).get("fact_id") or "")
            if fact_id:
                snapshots_by_id[fact_id] = snapshot
    candidates = list(window_candidates)
    for conflict in removed_group_candidates:
        side_ids = [side.fact_id for side in conflict.sides if side.fact_id]
        if side_ids and any(item in snapshots_by_id for item in side_ids):
            continue
        candidates.append(conflict)
    stale: list[FactConflict] = []
    for conflict in candidates:
        if conflict.status != "open":
            continue
        reason = _stale_reason(
            conflict,
            snapshots_by_id,
            live_pair_keys,
            compatible_pair_keys=compatible_pair_keys,
        )
        if not reason:
            continue
        if _skip_llm_stale(
            conflict,
            reason,
            llm_on=llm_on,
            llm_truncated=llm_truncated,
            judged_pair_keys=judged_pair_keys or set(),
        ):
            continue
        updated = conflict.model_copy(deep=True)
        updated.status = "resolved"
        updated.resolution = ConflictResolution(
            method="stale",
            notes=reason,
            resolved_at=format_instant(utc_now()),
            resolved_by="conflict_scan",
        )
        updater = getattr(adapter, "update_conflict", None)
        saved = None
        if callable(updater):
            try:
                saved = updater(updated, brain_id, if_status="open")
            except TypeError:
                current = None
                getter = getattr(adapter, "get_conflict_by_id", None)
                if callable(getter):
                    current = getter(conflict.id, brain_id)
                current = _conflict_from_row(current) if current is not None else None
                if current is not None and current.status != "open":
                    continue
                saved = updater(updated, brain_id)
        else:
            saved = updated
        if saved is None:
            continue
        stale.append(saved)
    return stale


def _reload_conflict(adapter: Any, brain_id: str, conflict_id: str) -> FactConflict | None:
    getter = getattr(adapter, "get_conflict_by_id", None)
    if not callable(getter):
        return None
    return _conflict_from_row(getter(conflict_id, brain_id))


def _save_conflict_if_unchanged(
    adapter: Any,
    brain_id: str,
    conflict: FactConflict,
    current: FactConflict | None,
) -> FactConflict | None:
    if current is None:
        saver = getattr(adapter, "save_conflict", None)
        if not callable(saver):
            return conflict
        saved = saver(conflict, brain_id)
        if isinstance(saved, FactConflict):
            return saved
        if isinstance(saved, dict):
            return conflict_from_document(saved)
        return conflict
    updater = getattr(adapter, "update_conflict", None)
    expected_status = current.status
    expected_method = None
    if expected_status == "resolved":
        expected_method = current.resolution.method or ""
    saved = None
    if callable(updater):
        try:
            kwargs: dict[str, Any] = {"if_status": expected_status}
            if expected_method is not None:
                kwargs["if_resolution_method"] = expected_method
            saved = updater(conflict, brain_id, **kwargs)
        except TypeError:
            try:
                saved = updater(conflict, brain_id, if_status=expected_status)
            except TypeError:
                saved = updater(conflict, brain_id)
    else:
        saver = getattr(adapter, "save_conflict", None)
        saved = saver(conflict, brain_id) if callable(saver) else conflict
    if saved is None:
        return _reload_conflict(adapter, brain_id, conflict.id)
    if isinstance(saved, FactConflict):
        return saved
    if isinstance(saved, dict):
        return conflict_from_document(saved)
    return conflict


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
    if isinstance(current, FactConflict) and current.status == "dismissed":
        return current
    if (
        isinstance(current, FactConflict)
        and current.status == "resolved"
        and (current.resolution.method or "") != "stale"
    ):
        return current
    if isinstance(current, FactConflict) and current.status == "open":
        conflict.detected_at = current.detected_at
        conflict.status = current.status
        conflict.resolution = current.resolution
    return _save_conflict_if_unchanged(adapter, brain_id, conflict, current)


def _optional_graph(graph: Any | None = None) -> Any | None:
    return graph


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
    if method == "verified_side" and status == "resolved" and not winning_fact_id:
        raise ValueError("winning_fact_id is required for method=verified_side")
    live_graph = _optional_graph(graph)
    if winning_fact_id:
        side_ids = {side.fact_id for side in conflict.sides}
        if winning_fact_id not in side_ids:
            raise ValueError("winning_fact_id must be one of the conflict sides")
        if apply_facts and method == "verified_side" and status == "resolved":
            outcome = apply_verified_side_to_facts(
                live_graph,
                brain_id,
                winning_fact_id=winning_fact_id,
                losing_fact_ids=[
                    side.fact_id
                    for side in conflict.sides
                    if side.fact_id != winning_fact_id
                ],
                actor=resolved_by,
                note=notes,
            )
            apply_status = getattr(outcome, "status", outcome)
            if apply_status != "applied":
                details = "; ".join(
                    getattr(outcome, "rollback_errors", ()) or ()
                )
                message = "fact verification failed"
                if getattr(outcome, "rollback_failed", False):
                    message = (
                        f"{message}; rollback failed"
                        + (f": {details}" if details else "")
                    )
                elif details:
                    message = f"{message}: {details}"
                if details or getattr(outcome, "rollback_failed", False):
                    logger.error(
                        "%s brain_id=%s conflict_id=%s",
                        message,
                        brain_id,
                        conflict_id,
                    )
                raise ValueError(message)
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


def resolve_changes_verification(method: str | None) -> bool:
    return (method or "manual") == "verified_side"


def resolve_conflict_authorized(
    adapter: Any,
    brain_id: str,
    conflict_id: str,
    *,
    authorized: bool,
    actor: str | None,
    has_verification: bool,
    winning_fact_id: str | None = None,
    notes: str | None = None,
    method: str = "manual",
    graph: Any | None = None,
) -> FactConflict:
    """REST/MCP entry: require verification scope when the resolve writes fact state."""
    if not authorized:
        raise PermissionError("Unauthorized")
    method = method or "manual"
    principal = (actor or "").strip() or None
    if resolve_changes_verification(method):
        if not has_verification or not principal:
            raise VerificationPermissionDenied("verification scope is required")
    return resolve_conflict(
        adapter,
        brain_id,
        conflict_id,
        winning_fact_id=winning_fact_id,
        notes=notes,
        method=method,
        resolved_by=principal or "system",
        graph=graph,
        apply_facts=True,
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


def maybe_detect_after_ingest(
    brain_id: str,
    fact_ids: list[str],
    subject_uuids: list[str] | None = None,
) -> None:
    if not conflict_detect_on_ingest_enabled():
        return
    try:
        from src.core.saving.ingest_write_buffer import writes_are_staged

        if writes_are_staged():
            return
    except Exception:
        pass
    ids = [item for item in fact_ids if item]
    subjects = [item for item in (subject_uuids or []) if item]
    if not ids and not subjects:
        return
    try:
        from src.workers.tasks.conflicts import detect_conflicts_incremental

        detect_conflicts_incremental.apply_async(
            args=[
                {
                    "brain_id": brain_id,
                    "fact_ids": ids,
                    "subject_uuids": subjects,
                }
            ]
        )
        return
    except Exception:
        pass
    from src.services.data.main import data_adapter
    from src.services.kg_agent.main import graph_adapter

    detect_conflicts(
        data_adapter,
        graph_adapter,
        brain_id,
        fact_ids=ids or None,
        subject_uuids=subjects or None,
    )
