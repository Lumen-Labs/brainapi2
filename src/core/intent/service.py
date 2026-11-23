from __future__ import annotations

from datetime import datetime
from typing import Any, Iterable

from src.constants.data import TextChunk
from src.constants.kg import IdentificationParams, Node, Predicate
from src.core.intent.brain import writer_from_token
from src.core.intent.format import clamp_page, render_check, render_facts
from src.core.intent.models import (
    CheckResult,
    CorrectResult,
    IntentFact,
    RecallResult,
    RememberResult,
    ResponseFormat,
)
from src.core.intent.parse import ParsedFact, lexical_score, parse_fact_text, parse_source
from src.core.intent.retire import retire_outgoing_matches
from src.core.intent.status import intent_fact_from_chunk, intent_fact_from_row
from src.core.saving.identity import stable_node_id, stable_relationship_id
from src.core.saving.source_passages import source_content_hash
from src.core.saving.validity import (
    SUPERSEDED_BY,
    VALID_FROM,
    VALID_TO,
    InvalidAsOf,
    format_instant,
    is_valid_at,
    optional_instant,
    parse_instant,
    stamp_valid_from,
    utc_now,
)
from src.core.verification.facts import (
    UNVERIFIED,
    verification_scope,
    verified_rank_boost,
)
from src.core.verification.service import FactNotFound, apply_verification_action


def _as_of_instant(value: Any) -> datetime | None:
    try:
        return optional_instant(value, name="as_of")
    except InvalidAsOf as exc:
        raise ValueError(str(exc)) from exc


def _upsert_node(graph: Any, name: str, labels: list[str], brain_id: str) -> Node:
    entity_type = (labels[0] if labels else "ENTITY") or "ENTITY"
    node = Node(
        uuid=stable_node_id(name, entity_type),
        name=name,
        labels=[entity_type],
        description=name,
    )
    getter = getattr(graph, "get_by_identification_params", None)
    if callable(getter):
        try:
            found = getter(
                IdentificationParams(name=name, entity_types=[entity_type]),
                brain_id=brain_id,
                entity_types=[entity_type],
            )
        except Exception:
            found = None
        if found and getattr(found, "uuid", None):
            return found
    added = graph.add_nodes([node], brain_id)
    if isinstance(added, list) and added:
        return added[0]
    return node


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


def _list_graph_facts(
    graph: Any,
    brain_id: str,
    *,
    include_rejected: bool,
    as_of: datetime | None,
) -> list[IntentFact]:
    rows: list[dict[str, Any]] = []
    skip = 0
    page_size = 200
    cap = 2000
    total = 0
    while skip < cap:
        page, total = graph.list_facts_by_verification(
            brain_id,
            include_rejected=include_rejected,
            limit=page_size,
            skip=skip,
        )
        if not page:
            break
        rows.extend(page)
        skip += len(page)
        if total and skip >= int(total):
            break
    ids = [str(row.get("uuid") or "") for row in rows]
    contradicted = _open_conflicts(
        getattr(graph, "data_adapter", None),
        brain_id,
        ids,
    )
    facts: list[IntentFact] = []
    for row in rows:
        fact = intent_fact_from_row(
            row,
            as_of=as_of,
            contradicted_by=contradicted.get(str(row.get("uuid") or ""), []),
        )
        if fact.status == "rejected" and not include_rejected:
            continue
        facts.append(fact)
    return facts


def _list_passage_facts(
    data: Any,
    brain_id: str,
    *,
    query: str,
    include_rejected: bool,
    as_of: datetime | None,
) -> list[IntentFact]:
    getter = getattr(data, "get_text_chunks", None)
    if not callable(getter):
        return []
    try:
        chunks, _total = getter(
            brain_id=brain_id,
            limit=80,
            skip=0,
            query_text=query or None,
        )
    except TypeError:
        chunks, _total = getter(brain_id=brain_id, limit=80, skip=0)
    except Exception:
        return []
    facts: list[IntentFact] = []
    for chunk in chunks or []:
        fact = intent_fact_from_chunk(chunk, as_of=as_of)
        if fact.status == "rejected" and not include_rejected:
            continue
        if query and lexical_score(query, fact.text) <= 0 and query.lower() not in (fact.text or "").lower():
            continue
        facts.append(fact)
    return facts


def _in_window(fact: IntentFact, as_of: datetime | None) -> bool:
    fields = {
        VALID_FROM: fact.valid_at,
        VALID_TO: fact.invalid_at,
        "invalid_at": fact.invalid_at,
        SUPERSEDED_BY: fact.superseded_by,
    }
    instant = as_of if as_of is not None else utc_now()
    return is_valid_at(fields, instant, deprecated=False)


def _started_by(fact: IntentFact, as_of: datetime | None) -> bool:
    if as_of is None or not fact.valid_at:
        return True
    start = parse_instant(fact.valid_at)
    return start is None or start <= as_of


def _same_triple(parsed: ParsedFact, fact: IntentFact) -> bool:
    if not fact.subject or not parsed.subject or not fact.predicate:
        return False
    if fact.subject.lower() != parsed.subject.lower():
        return False
    return parsed.predicate == fact.predicate


def _same_object(parsed: ParsedFact, fact: IntentFact) -> bool:
    if not fact.object or not parsed.object:
        return False
    return fact.object.lower() == parsed.object.lower()


def _rank_key(fact: IntentFact, query: str) -> tuple:
    score = lexical_score(query, " ".join((fact.text, fact.subject, fact.object, fact.predicate)))
    boost = verified_rank_boost() if fact.status == "verified" else 0.0
    current = 0 if fact.status in {"verified", "unverified"} else 1
    return (-(score + boost), current, fact.fact_id)


def _merge_facts(*groups: Iterable[IntentFact]) -> list[IntentFact]:
    merged: dict[str, IntentFact] = {}
    for group in groups:
        for fact in group:
            key = fact.fact_id or f"{fact.kind}:{fact.natural_id()}"
            previous = merged.get(key)
            if previous is None or (previous.kind == "passage" and fact.kind != "passage"):
                merged[key] = fact
    return list(merged.values())


def _current_source_chunks(data: Any, brain_id: str, source_key: str) -> list[Any]:
    getter = getattr(data, "get_text_chunks", None)
    if not callable(getter) or not source_key:
        return []
    try:
        chunks, _total = getter(
            brain_id=brain_id,
            limit=80,
            skip=0,
            metadata_eq={"source_key": source_key},
        )
    except TypeError:
        return []
    except Exception:
        return []
    live = []
    for chunk in chunks or []:
        meta = getattr(chunk, "metadata", None) or {}
        if meta.get("superseded_at") or meta.get("valid_to"):
            continue
        live.append(chunk)
    return live


def _save_passage(
    data: Any,
    *,
    brain_id: str,
    text: str,
    meta: dict[str, Any],
    valid_from: str,
    fact_id: str,
) -> tuple[TextChunk | None, list[str]]:
    if data is None:
        return None, []
    meta_keys = dict(meta)
    meta_keys.setdefault("fact_id", fact_id)
    source_key = str(meta_keys.get("source_key") or "").strip()
    digest = source_content_hash(text)
    meta_keys["source_content_hash"] = digest
    if source_key:
        current = _current_source_chunks(data, brain_id, source_key)
        same = [
            chunk
            for chunk in current
            if (getattr(chunk, "metadata", None) or {}).get("source_content_hash") == digest
        ]
        if same:
            return same[0], []
        chunk = TextChunk(text=text, metadata=dict(meta_keys))
        saved = data.save_text_chunk(chunk, brain_id)
        stale = [str(item.id) for item in current if getattr(item, "id", None)]
        marker = getattr(data, "mark_text_chunks_superseded", None)
        if stale and callable(marker):
            marker(
                stale,
                valid_from,
                brain_id,
                superseded_by=getattr(saved, "id", None),
                valid_to=valid_from,
            )
        return saved, stale
    chunk = TextChunk(text=text, metadata=dict(meta_keys))
    return data.save_text_chunk(chunk, brain_id), []


def _write_graph_fact(
    graph: Any,
    parsed: ParsedFact,
    *,
    brain_id: str,
    properties: dict[str, Any],
) -> tuple[Node, Predicate, Node]:
    subject = _upsert_node(graph, parsed.subject, [parsed.subject_type], brain_id)
    obj = _upsert_node(graph, parsed.object, [parsed.object_type], brain_id)
    valid_from = properties.get(VALID_FROM)
    rel_uuid = stable_relationship_id(
        subject.uuid,
        parsed.predicate,
        obj.uuid,
        str(valid_from or properties.get("source_key") or ""),
    )
    predicate = Predicate(
        uuid=rel_uuid,
        name=parsed.predicate,
        description=parsed.text,
        direction="out",
        properties=dict(properties),
        verification_state=UNVERIFIED,
    )
    graph.add_relationship(subject, predicate, obj, brain_id)
    return subject, predicate, obj


def remember_fact(
    *,
    fact: str,
    source: str,
    brain_id: str,
    token: str | None = None,
    valid_from: str | None = None,
    source_key: str | None = None,
    parsed: ParsedFact | None = None,
    graph: Any,
    data: Any | None = None,
) -> RememberResult:
    parsed = parsed or parse_fact_text(fact)
    source_fields = parse_source(source, source_key)
    if not source_fields.get("doc") and not source_fields.get("url"):
        raise ValueError("source is required (document name or URL)")
    try:
        start = optional_instant(valid_from, name="valid_from") if valid_from else utc_now()
    except InvalidAsOf as exc:
        raise ValueError(str(exc)) from exc
    stamp = format_instant(start)
    writer = writer_from_token(token, brain_id)
    props = stamp_valid_from(
        {
            "fact_text": parsed.text,
            "writer": writer,
            "source": source.strip(),
            "source_doc": source_fields.get("doc"),
            "source_url": source_fields.get("url"),
            "source_key": source_fields.get("source_key"),
            "verification_state": UNVERIFIED,
            "subject_name": parsed.subject,
            "object_name": parsed.object,
        },
        stamp,
    )
    subject, predicate, obj = _write_graph_fact(
        graph, parsed, brain_id=brain_id, properties=props
    )
    chunk_meta = {
        **props,
        "subject": parsed.subject,
        "predicate": parsed.predicate,
        "object": parsed.object,
        "fact_id": predicate.uuid,
    }
    _saved_chunk, superseded_chunks = _save_passage(
        data,
        brain_id=brain_id,
        text=parsed.text,
        meta=chunk_meta,
        valid_from=stamp,
        fact_id=predicate.uuid,
    )
    if _saved_chunk is not None:
        chunk_ids = [str(_saved_chunk.id)]
        graph.update_properties(
            predicate.uuid,
            "relationship",
            brain_id=brain_id,
            new_properties={"source_chunk_ids": chunk_ids},
        )
    closed = retire_outgoing_matches(
        graph,
        brain_id=brain_id,
        subject_uuid=subject.uuid,
        predicate_name=parsed.predicate,
        keep_uuid=predicate.uuid,
        reason="single-valued predicate replaced",
        closed_at=stamp,
        superseded_by=predicate.uuid,
        superseded_chunk_ids=superseded_chunks,
        require_functional_or_source=True,
    )
    row = graph.get_fact_record(predicate.uuid, brain_id) or {
        "kind": "relationship",
        "uuid": predicate.uuid,
        "name": parsed.predicate,
        "description": parsed.text,
        "properties": {**props, "uuid": predicate.uuid},
        "subject_uuid": subject.uuid,
        "subject_name": subject.name,
        "subject_labels": subject.labels,
        "object_uuid": obj.uuid,
        "object_name": obj.name,
        "object_labels": obj.labels,
    }
    intent = intent_fact_from_row(row)
    heading = f"Remembered unverified fact: {intent.natural_id()}"
    return RememberResult(
        text=heading,
        brain_id=brain_id,
        fact=intent,
        superseded=closed,
    )


def recall_facts(
    *,
    question: str,
    brain_id: str,
    as_of: str | None = None,
    response_format: ResponseFormat = "concise",
    include_rejected: bool = False,
    limit: int | None = None,
    skip: int = 0,
    graph: Any,
    data: Any | None = None,
) -> RecallResult:
    if not (question or "").strip():
        raise ValueError("question is required")
    instant = _as_of_instant(as_of)
    page = clamp_page(limit, response_format)
    offset = max(0, int(skip or 0))
    with verification_scope(include_rejected=True):
        graph_all = _list_graph_facts(
            graph, brain_id, include_rejected=True, as_of=instant
        )
        graph_facts = [
            fact for fact in graph_all if fact.status != "rejected" or include_rejected
        ]
        rejected_ids = {fact.fact_id for fact in graph_all if fact.status == "rejected"}
        if data is not None:
            contradicted = _open_conflicts(
                data, brain_id, [fact.fact_id for fact in graph_facts]
            )
            for fact in graph_facts:
                extra = contradicted.get(fact.fact_id) or []
                if extra:
                    fact.contradicted_by = list(dict.fromkeys([*fact.contradicted_by, *extra]))
        passages = _list_passage_facts(
            data,
            brain_id,
            query=question,
            include_rejected=include_rejected,
            as_of=instant,
        ) if data is not None else []
    merged = _merge_facts(graph_facts, passages)
    merged = [
        fact
        for fact in merged
        if (fact.fact_id not in rejected_ids or include_rejected)
        and (fact.status != "rejected" or include_rejected)
        and _in_window(fact, instant)
    ]
    scored = [fact for fact in merged if lexical_score(question, " ".join((fact.text, fact.subject, fact.object, fact.natural_id()))) > 0 or question.lower() in fact.text.lower()]
    if not scored:
        scored = merged
    scored.sort(key=lambda fact: _rank_key(fact, question))
    page_items = scored[offset : offset + page]
    heading = f"What's true for: {question.strip()}"
    if instant is not None:
        heading = f"{heading} (as of {format_instant(instant)})"
    text = render_facts(page_items, response_format=response_format, heading=heading)
    next_skip = offset + page if offset + page < len(scored) else None
    return RecallResult(
        text=text,
        as_of=format_instant(instant) if instant else None,
        brain_id=brain_id,
        facts=page_items,
        total=len(scored),
        skip=offset,
        limit=page,
        next_skip=next_skip,
        truncated=bool(next_skip is not None) or text.endswith("… truncated"),
    )


def _claim_matches(claim: str, fact: IntentFact) -> bool:
    parsed = parse_fact_text(claim)
    return _same_triple(parsed, fact)


def check_claim(
    *,
    claim: str,
    brain_id: str,
    as_of: str | None = None,
    graph: Any,
    data: Any | None = None,
    response_format: ResponseFormat = "concise",
) -> CheckResult:
    if not (claim or "").strip():
        raise ValueError("claim is required")
    instant = _as_of_instant(as_of)
    parsed = parse_fact_text(claim)
    with verification_scope(include_rejected=True):
        facts = _list_graph_facts(graph, brain_id, include_rejected=True, as_of=instant)
        if data is not None:
            contradicted = _open_conflicts(data, brain_id, [fact.fact_id for fact in facts])
            for fact in facts:
                extra = contradicted.get(fact.fact_id) or []
                if extra:
                    fact.contradicted_by = list(dict.fromkeys([*fact.contradicted_by, *extra]))
    matches = [
        fact
        for fact in facts
        if _same_triple(parsed, fact) and _started_by(fact, instant)
    ]
    matches.sort(key=lambda fact: _rank_key(fact, claim))
    verdict: str = "unknown"
    evidence: list[IntentFact] = []
    for fact in matches:
        same_object = _same_object(parsed, fact)
        historically_current = _in_window(fact, instant)
        if fact.status == "rejected" and same_object:
            verdict = "contradicted"
            evidence = [fact]
            break
        if historically_current and fact.status != "rejected":
            if not same_object:
                verdict = "contradicted"
                evidence = [fact]
                break
            if fact.contradicted_by and instant is None:
                verdict = "contradicted"
                evidence = [fact]
                break
            verdict = "supported"
            evidence = [fact]
            break
        if fact.status == "superseded" and same_object:
            if verdict == "unknown":
                verdict = "superseded"
                evidence = [fact]
            continue
    if verdict == "superseded" and not evidence:
        evidence = matches[:3]
    if verdict == "unknown":
        evidence = matches[:3]
    result = CheckResult(
        text="",
        verdict=verdict,  # type: ignore[arg-type]
        claim=claim.strip(),
        as_of=format_instant(instant) if instant else None,
        brain_id=brain_id,
        evidence=evidence,
    )
    result.text = render_check(result, response_format=response_format)
    return result


def _load_target(
    graph: Any,
    brain_id: str,
    fact_id: str | None,
    text: str | None,
) -> dict[str, Any]:
    if fact_id:
        row = graph.get_fact_record(fact_id, brain_id)
        if row is None:
            raise FactNotFound(fact_id)
        return row
    query = (text or "").strip()
    if not query:
        raise ValueError("pass fact_id or the fact text")
    parsed = parse_fact_text(query)
    with verification_scope(include_rejected=True):
        facts = _list_graph_facts(graph, brain_id, include_rejected=True, as_of=None)
    exact = [
        fact
        for fact in facts
        if fact.text.lower() == query.lower() or fact.natural_id().lower() == query.lower()
    ]
    structural = [
        fact
        for fact in facts
        if _same_triple(parsed, fact) and _same_object(parsed, fact)
    ]
    matches = exact or structural
    current = [fact for fact in matches if fact.status in {"verified", "unverified"}]
    pool = current or matches
    if not pool:
        raise FactNotFound(query)
    if len(pool) != 1:
        raise ValueError("multiple facts match that text; pass fact_id")
    chosen = pool[0]
    row = graph.get_fact_record(chosen.fact_id, brain_id)
    if row is None:
        raise FactNotFound(chosen.fact_id)
    return row


def correct_fact(
    *,
    reason: str,
    brain_id: str,
    token: str | None = None,
    fact_id: str | None = None,
    text: str | None = None,
    correction: str | None = None,
    graph: Any,
    data: Any | None = None,
) -> CorrectResult:
    reason_text = (reason or "").strip()
    if not reason_text:
        raise ValueError("reason is required")
    row = _load_target(graph, brain_id, fact_id, text)
    previous = intent_fact_from_row(row)
    writer = writer_from_token(token, brain_id)
    stamp = format_instant(utc_now())
    correction_text = (correction or "").strip()
    if (
        correction_text
        and previous.status == "superseded"
        and previous.superseded_by
    ):
        existing_row = graph.get_fact_record(previous.superseded_by, brain_id)
        if existing_row is not None:
            existing = intent_fact_from_row(existing_row)
            wanted = parse_fact_text(correction_text)
            if previous.subject:
                wanted.subject = previous.subject
            if previous.predicate:
                wanted.predicate = previous.predicate
            if _same_triple(wanted, existing) and _same_object(wanted, existing):
                return CorrectResult(
                    text=f"Already corrected {previous.natural_id()} → {existing.natural_id()}",
                    brain_id=brain_id,
                    fact=existing,
                    previous=previous,
                    superseded=[previous.fact_id],
                )
    if not correction_text:
        record = apply_verification_action(
            graph,
            fact_id=previous.fact_id,
            brain_id=brain_id,
            action="reject",
            verified_by=writer,
            verification_source=previous.source.doc or previous.source.url,
            note=reason_text,
        )
        closed = graph.get_fact_record(previous.fact_id, brain_id) or row
        props = _properties_map(closed)
        if not props.get("retired_reason"):
            graph.update_properties(
                previous.fact_id,
                closed.get("kind") or "relationship",
                brain_id=brain_id,
                new_properties={"retired_reason": reason_text},
            )
            closed = graph.get_fact_record(previous.fact_id, brain_id) or closed
        updated = intent_fact_from_row(closed)
        heading = f"Rejected {updated.natural_id()} ({reason_text})"
        return CorrectResult(
            text=heading,
            brain_id=brain_id,
            fact=updated,
            previous=previous,
            superseded=[updated.fact_id],
        )

    parsed = parse_fact_text(correction_text)
    if previous.subject and parsed.subject.lower() != previous.subject.lower():
        parsed.subject = previous.subject
        parsed.subject_type = (row.get("subject_labels") or ["ENTITY"])[0]
    if previous.predicate:
        parsed.predicate = previous.predicate
    remembered = remember_fact(
        fact=correction_text,
        parsed=parsed,
        source=previous.source.url or previous.source.doc or previous.source.source_key or "correction",
        brain_id=brain_id,
        token=token,
        valid_from=stamp,
        source_key=previous.source.source_key,
        graph=graph,
        data=data,
    )
    keep_uuid = remembered.fact.fact_id if remembered.fact else None
    closed = retire_outgoing_matches(
        graph,
        brain_id=brain_id,
        subject_uuid=str(row.get("subject_uuid") or ""),
        predicate_name=previous.predicate or parsed.predicate,
        keep_uuid=keep_uuid,
        reason=reason_text,
        closed_at=stamp,
        superseded_by=keep_uuid,
        require_functional_or_source=True,
        always_uuids=[previous.fact_id],
    )
    if previous.fact_id not in closed and previous.fact_id != keep_uuid:
        graph.update_properties(
            previous.fact_id,
            row.get("kind") or "relationship",
            brain_id=brain_id,
            new_properties={
                "invalid_at": stamp,
                "deprecated": True,
                "valid_to": stamp,
                SUPERSEDED_BY: remembered.fact.fact_id,
                "retired_reason": reason_text,
            },
        )
        closed.append(previous.fact_id)
    updated_previous = intent_fact_from_row(graph.get_fact_record(previous.fact_id, brain_id) or row)
    heading = (
        f"Corrected {updated_previous.natural_id()} → {remembered.fact.natural_id()} "
        f"({reason_text})"
    )
    return CorrectResult(
        text=heading,
        brain_id=brain_id,
        fact=remembered.fact,
        previous=updated_previous,
        superseded=list(dict.fromkeys([*closed, *remembered.superseded])),
    )
