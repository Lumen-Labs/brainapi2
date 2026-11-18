from __future__ import annotations

from typing import Literal, Optional

from fastapi import HTTPException

from src.core.verification.facts import (
    REJECTED,
    UNVERIFIED,
    VERIFIED,
    VERIFICATION_STATES,
    FactRecord,
    VerificationState,
    append_history_entry,
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
    verified_by: str | None = None,
    verification_source: str | None = None,
    note: str | None = None,
) -> FactRecord:
    fact_id = str(fact_id or "").strip()
    if not fact_id:
        raise HTTPException(status_code=400, detail="fact_id is required")

    target_state: VerificationState
    if action == "verify":
        target_state = VERIFIED
        if not (verified_by or "").strip():
            raise HTTPException(
                status_code=400,
                detail="verified_by is required when verifying a fact",
            )
    elif action == "reject":
        target_state = REJECTED
        if not (verified_by or "").strip():
            raise HTTPException(
                status_code=400,
                detail="verified_by is required when rejecting a fact",
            )
    elif action == "reset":
        target_state = UNVERIFIED
    else:
        raise HTTPException(status_code=400, detail="Unknown verification action")

    current = graph.get_fact_record(fact_id, brain_id)
    if current is None:
        raise FactNotFound(fact_id)

    current_record = record_from_graph_row(current)
    actor = (verified_by or "").strip() or None
    source = (verification_source or "").strip() or None
    comment = (note or "").strip() or None
    verified_at = utc_now_iso() if target_state != UNVERIFIED else None
    if action == "reset":
        actor = actor or current_record.verification.verified_by
        verified_at = utc_now_iso()

    history = append_history_entry(
        current_record.history,
        from_state=current_record.verification.state,
        to_state=target_state,
        verified_by=actor,
        verified_at=verified_at or utc_now_iso(),
        verification_source=source,
        note=comment,
    )
    properties = verification_write_properties(
        state=target_state,
        verified_by=None if target_state == UNVERIFIED else actor,
        verified_at=None if target_state == UNVERIFIED else verified_at,
        verification_source=None if target_state == UNVERIFIED else source,
        note=None if target_state == UNVERIFIED else comment,
        history=history,
    )
    updated = graph.set_fact_verification(
        fact_id,
        brain_id,
        kind=current_record.kind,
        properties=properties,
    )
    if updated is None:
        raise FactNotFound(fact_id)
    return record_from_graph_row(updated)


def get_fact_record(graph, fact_id: str, brain_id: str) -> FactRecord:
    row = graph.get_fact_record(fact_id, brain_id)
    if row is None:
        raise FactNotFound(fact_id)
    return record_from_graph_row(row)


def list_fact_records(
    graph,
    brain_id: str,
    *,
    state: VerificationState | None = None,
    include_rejected: bool = False,
    limit: int = 50,
    skip: int = 0,
) -> tuple[list[FactRecord], int]:
    rows, total = graph.list_facts_by_verification(
        brain_id,
        state=state,
        include_rejected=include_rejected if state != REJECTED else True,
        limit=limit,
        skip=skip,
    )
    return [record_from_graph_row(row) for row in rows], int(total or 0)
