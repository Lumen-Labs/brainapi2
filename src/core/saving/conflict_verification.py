"""Adapter seam onto PR #113 fact verification.

Conflict sides do not own verification state. They read it from
``src.core.verification`` when that package is present, and
``resolve_conflict(..., method="verified_side")`` asks #113 to verify the
winning fact and reject the other side. ``resolve_by_verified_fact`` only
closes open conflicts for a fact that #113 has already verified.

Until #113 merges, these helpers no-op and sides stay ``unverified``.
"""

from __future__ import annotations

from typing import Any, Callable


def read_fact_verification_state(
    fact_id: str,
    *,
    graph: Any | None = None,
    brain_id: str | None = None,
    entity: Any | None = None,
) -> str | None:
    """Return #113's state for a fact, or ``None`` if that API is absent."""
    if graph is not None and brain_id and fact_id:
        try:
            from src.core.verification.service import get_fact_record

            record = get_fact_record(graph, fact_id, brain_id)
            state = getattr(getattr(record, "verification", None), "state", None)
            if state:
                return str(state)
        except Exception:
            pass
    if entity is not None:
        try:
            from src.core.verification.facts import extract_verification

            return str(extract_verification(entity).state)
        except Exception:
            props = getattr(entity, "properties", None)
            if isinstance(props, dict):
                raw = str(props.get("verification_state") or "").strip().lower()
                if raw in {"unverified", "verified", "rejected"}:
                    return raw
            raw = str(getattr(entity, "verification_state", "") or "").strip().lower()
            if raw in {"unverified", "verified", "rejected"}:
                return raw
    return None


def apply_fact_verification_action(
    graph: Any | None,
    *,
    fact_id: str,
    brain_id: str,
    action: str,
    verified_by: str | None = None,
    note: str | None = None,
) -> bool:
    """Ask #113 to verify or reject a fact. Returns False when #113 is absent."""
    if graph is None or action not in {"verify", "reject"}:
        return False
    try:
        from src.core.verification.service import apply_verification_action
    except ImportError:
        return False
    try:
        apply_verification_action(
            graph,
            fact_id=fact_id,
            brain_id=brain_id,
            action=action,  # type: ignore[arg-type]
            verified_by=verified_by or "conflict",
            verification_source="conflict",
            note=note,
        )
    except Exception:
        return False
    return True


def apply_verified_side_to_facts(
    graph: Any | None,
    brain_id: str,
    *,
    winning_fact_id: str,
    losing_fact_ids: list[str],
    verified_by: str | None = None,
    note: str | None = None,
    apply_action: Callable[..., bool] | None = None,
) -> bool:
    """Verify the winning fact and reject the other conflict side via #113."""
    apply = apply_action or apply_fact_verification_action
    applied = apply(
        graph,
        fact_id=winning_fact_id,
        brain_id=brain_id,
        action="verify",
        verified_by=verified_by,
        note=note,
    )
    for fact_id in losing_fact_ids:
        apply(
            graph,
            fact_id=fact_id,
            brain_id=brain_id,
            action="reject",
            verified_by=verified_by,
            note=note,
        )
    return applied


def hydrate_conflict_sides(
    conflict: Any,
    *,
    graph: Any | None = None,
    brain_id: str | None = None,
    read_state: Callable[..., str] | None = None,
) -> Any:
    """Overwrite each side's displayed state from #113 when that API answers."""
    if conflict is None:
        return conflict
    reader = read_state or read_fact_verification_state
    for side in getattr(conflict, "sides", []) or []:
        state = reader(
            side.fact_id,
            graph=graph,
            brain_id=brain_id,
        )
        if state:
            side.verification_state = state
    return conflict
