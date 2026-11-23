from __future__ import annotations

from datetime import datetime
from typing import Any

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
from src.core.intent.lock import subject_write_lock
from src.core.intent.parse import ParsedFact, parse_fact_text, parse_source
from src.core.intent.retire import apply_open_update, retire_outgoing_matches
from src.core.intent.retrieve import collect_intent_facts, fact_in_window, rank_key
from src.core.intent.status import intent_fact_from_row
from src.core.saving.identity import stable_node_id, stable_relationship_id
from src.core.saving.source_passages import source_content_hash
from src.core.saving.validity import (
    SUPERSEDED_BY,
    VALID_FROM,
    InvalidAsOf,
    format_instant,
    optional_instant,
    parse_instant,
    stamp_valid_from,
    utc_now,
)
from src.core.verification.facts import UNVERIFIED
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
    lock_uuid = stable_node_id(parsed.subject, parsed.subject_type)
    with subject_write_lock(brain_id, lock_uuid, parsed.predicate):
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
        retirement = retire_outgoing_matches(
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
        closed = list(retirement.closed)
        if retirement.successor_start:
            incoming_start = parse_instant(stamp)
            successor_start = parse_instant(retirement.successor_start)
            if incoming_start is None or successor_start is None or successor_start >= incoming_start:
                apply_open_update(
                    graph,
                    predicate.uuid,
                    brain_id,
                    {
                        "invalid_at": retirement.successor_start,
                        "deprecated": True,
                        "valid_to": retirement.successor_start,
                        SUPERSEDED_BY: retirement.successor_uuid,
                        "retired_reason": "recorded as historical before a later fact",
                    },
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
    scored, _distances = collect_intent_facts(
        query=question,
        brain_id=brain_id,
        graph=graph,
        data=data,
        as_of=instant,
        include_rejected=include_rejected,
    )
    matched = bool(scored)
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
        matched=matched,
        total=len(scored),
        skip=offset,
        limit=page,
        next_skip=next_skip,
        truncated=bool(next_skip is not None) or text.endswith("… truncated"),
    )


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
    facts, _distances = collect_intent_facts(
        query=claim,
        brain_id=brain_id,
        graph=graph,
        data=data,
        as_of=instant,
        include_rejected=True,
    )
    matches = [
        fact
        for fact in facts
        if _same_triple(parsed, fact) and _started_by(fact, instant)
    ]
    matches.sort(key=lambda fact: rank_key(fact, claim))
    verdict: str = "unknown"
    evidence: list[IntentFact] = []
    for fact in matches:
        same_object = _same_object(parsed, fact)
        historically_current = fact_in_window(fact, instant)
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
    if verdict == "unknown":
        evidence = []
    elif verdict == "superseded" and not evidence:
        evidence = matches[:3]
    result = CheckResult(
        text="",
        verdict=verdict,  # type: ignore[arg-type]
        claim=claim.strip(),
        as_of=format_instant(instant) if instant else None,
        brain_id=brain_id,
        evidence=evidence,
        matched=bool(matches),
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
    facts, _distances = collect_intent_facts(
        query=query,
        brain_id=brain_id,
        graph=graph,
        data=None,
        as_of=None,
        include_rejected=True,
    )
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
    lock_uuid = str(row.get("subject_uuid") or "")
    with subject_write_lock(brain_id, lock_uuid, previous.predicate or ""):
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
            apply_verification_action(
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
        retirement = retire_outgoing_matches(
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
        closed = list(retirement.closed)
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
