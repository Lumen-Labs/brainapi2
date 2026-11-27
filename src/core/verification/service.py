from __future__ import annotations

from typing import Literal, Optional

from fastapi import HTTPException

from src.core.verification.facts import (
    REJECTED,
    UNVERIFIED,
    VERIFIED,
    VERIFICATION_STATES,
    FactHistoryEntry,
    FactRecord,
    VerificationState,
    attach_chunk_listing_metadata,
    fact_record_matches_listing_filters,
    listing_chunk_ids_for_filters,
    node_is_verifiable_event,
    normalize_listing_filter,
    normalize_state,
    record_from_graph_row,
    utc_now_iso,
    verification_write_properties,
)

VerificationAction = Literal["verify", "reject", "reset"]


class FactNotFound(LookupError):
    """Raised when a fact UUID is not present in the brain."""


def require_state(value: str | None) -> VerificationState | None:
    if value is None or value == "" or value == "all":
        return None
    normalized = normalize_state(value)
    if value.strip().lower() not in VERIFICATION_STATES:
        raise HTTPException(
            status_code=400,
            detail=(
                "verification state must be one of: "
                + ", ".join(VERIFICATION_STATES)
            ),
        )
    return normalized


def apply_verification_action(
    graph,
    *,
    fact_id: str,
    brain_id: str,
    action: VerificationAction,
    actor: str | None = None,
    verified_by: str | None = None,
    verification_source: str | None = None,
    note: str | None = None,
) -> FactRecord:
    fact_id = str(fact_id or "").strip()
    if not fact_id:
        raise HTTPException(status_code=400, detail="fact_id is required")

    principal = (actor or verified_by or "").strip() or None
    target_state: VerificationState
    if action == "verify":
        target_state = VERIFIED
        if not principal:
            raise HTTPException(
                status_code=400,
                detail="authenticated principal is required when verifying a fact",
            )
    elif action == "reject":
        target_state = REJECTED
        if not principal:
            raise HTTPException(
                status_code=400,
                detail="authenticated principal is required when rejecting a fact",
            )
    elif action == "reset":
        target_state = UNVERIFIED
    else:
        raise HTTPException(status_code=400, detail="Unknown verification action")

    current = graph.get_fact_record(fact_id, brain_id)
    if current is None:
        raise FactNotFound(fact_id)

    current_record = record_from_graph_row(current)
    if current_record.kind == "node" and not node_is_verifiable_event(
        current_record.labels
    ):
        raise FactNotFound(fact_id)
    source = (verification_source or "").strip() or None
    comment = (note or "").strip() or None
    verified_at = utc_now_iso()
    if action == "reset":
        principal = principal or current_record.verification.verified_by

    history_entry = FactHistoryEntry(
        from_state=current_record.verification.state,
        to_state=target_state,
        verified_by=principal,
        verified_at=verified_at,
        verification_source=source,
        note=comment,
    )
    properties = verification_write_properties(
        state=target_state,
        verified_by=None if target_state == UNVERIFIED else principal,
        verified_at=None if target_state == UNVERIFIED else verified_at,
        verification_source=None if target_state == UNVERIFIED else source,
        note=None if target_state == UNVERIFIED else comment,
        history=[],
    )
    properties.pop("verification_history", None)
    updated = graph.set_fact_verification(
        fact_id,
        brain_id,
        kind=current_record.kind,
        properties=properties,
        history_entry=history_entry,
    )
    if updated is None:
        raise FactNotFound(fact_id)
    return attach_chunk_listing_metadata(
        [record_from_graph_row(updated)], brain_id
    )[0]


def get_fact_record(graph, fact_id: str, brain_id: str) -> FactRecord:
    row = graph.get_fact_record(fact_id, brain_id)
    if row is None:
        raise FactNotFound(fact_id)
    return attach_chunk_listing_metadata(
        [record_from_graph_row(row)], brain_id
    )[0]


def _list_all_fact_rows(graph, brain_id: str, **kwargs) -> list[dict]:
    skip = 0
    page = 500
    collected: list[dict] = []
    while True:
        rows, total = graph.list_facts_by_verification(
            brain_id,
            skip=skip,
            limit=page,
            **kwargs,
        )
        collected.extend(rows or [])
        skip += len(rows or [])
        if not rows or skip >= int(total or 0):
            break
    return collected


def list_fact_records(
    graph,
    brain_id: str,
    *,
    state: VerificationState | None = None,
    include_rejected: bool = False,
    limit: int = 50,
    skip: int = 0,
    interview_id: str | None = None,
    topic_id: str | None = None,
    entity_id: str | None = None,
    source: str | None = None,
) -> tuple[list[FactRecord], int]:
    interview_id = normalize_listing_filter(interview_id)
    topic_id = normalize_listing_filter(topic_id)
    source = normalize_listing_filter(source)
    post_filter = bool(interview_id or topic_id or source)
    graph_kwargs = {
        "state": state,
        "include_rejected": include_rejected if state != REJECTED else True,
        "entity_id": entity_id,
        "interview_id": None if post_filter else interview_id,
        "topic_id": None if post_filter else topic_id,
        "source": None if post_filter else source,
    }
    if post_filter:
        rows = _list_all_fact_rows(graph, brain_id, **graph_kwargs)
        graph_total = len(rows)
    else:
        rows, graph_total = graph.list_facts_by_verification(
            brain_id,
            limit=limit,
            skip=skip,
            **graph_kwargs,
        )
    records = attach_chunk_listing_metadata(
        [record_from_graph_row(row) for row in rows],
        brain_id,
    )
    if post_filter:
        matching = listing_chunk_ids_for_filters(
            brain_id,
            interview_id=interview_id,
            topic_id=topic_id,
            source=source,
        )
        records = [
            record
            for record in records
            if fact_record_matches_listing_filters(
                record,
                interview_id=interview_id,
                topic_id=topic_id,
                source=source,
                interview_chunk_ids=matching["interview"],
                topic_chunk_ids=matching["topic"],
                source_chunk_ids=matching["source"],
            )
        ]
        total = len(records)
        records = records[skip : skip + limit]
        return records, total
    return records, int(graph_total or 0)
