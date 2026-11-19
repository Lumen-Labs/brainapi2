"""Adapter seam onto PR #113 fact verification.

Conflict sides do not own verification state. They read it from
``src.core.verification``, and ``resolve_conflict(..., method="verified_side")``
asks #113 to verify the winning fact and reject the other side.
``resolve_by_verified_fact`` only closes open conflicts for a fact that #113
has already verified. #113 verify endpoints call that hook after they persist.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Literal

VerificationApplyResult = Literal["applied", "absent", "failed"]
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VerifiedSideApplyResult:
    status: VerificationApplyResult
    rollback_failed: bool = False
    rollback_errors: tuple[str, ...] = ()

    def __eq__(self, other: object) -> bool:
        if isinstance(other, str):
            return self.status == other
        if isinstance(other, VerifiedSideApplyResult):
            return (
                self.status == other.status
                and self.rollback_failed == other.rollback_failed
                and self.rollback_errors == other.rollback_errors
            )
        return NotImplemented


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
    actor: str | None = None,
    verified_by: str | None = None,
    note: str | None = None,
) -> VerificationApplyResult:
    """Ask #113 to verify, reject, or reset a fact.

    ``absent`` means the verification package is not imported or there is no
    graph to call. ``failed`` means #113 is present and the action raised.
    Identity comes from ``actor`` (the authenticated principal). ``verified_by``
    is only a fallback for older internal callers.
    """
    if action not in {"verify", "reject", "reset"}:
        return "failed"
    if graph is None:
        return "absent"
    try:
        from src.core.verification.service import apply_verification_action
    except ImportError:
        return "absent"
    principal = (actor or verified_by or "").strip() or None
    try:
        apply_verification_action(
            graph,
            fact_id=fact_id,
            brain_id=brain_id,
            action=action,  # type: ignore[arg-type]
            actor=principal,
            verification_source="conflict",
            note=note,
        )
    except Exception:
        return "failed"
    return "applied"


def _restore_action_for_prior_state(state: str | None) -> str:
    normalized = str(state or "").strip().lower()
    if normalized == "verified":
        return "verify"
    if normalized == "rejected":
        return "reject"
    return "reset"


def apply_verified_side_to_facts(
    graph: Any | None,
    brain_id: str,
    *,
    winning_fact_id: str,
    losing_fact_ids: list[str],
    actor: str | None = None,
    verified_by: str | None = None,
    note: str | None = None,
    apply_action: Callable[..., VerificationApplyResult] | None = None,
    read_state: Callable[..., str | None] | None = None,
) -> VerifiedSideApplyResult:
    """Verify the winning fact and reject the other conflict side via #113.

    Snapshot prior verification states before writing. If a later write fails,
    restore every mutated fact to that snapshot. The caller must not mark the
    conflict resolved unless this returns ``applied``.
    """
    apply = apply_action or apply_fact_verification_action
    reader = read_state or read_fact_verification_state
    principal = (actor or verified_by or "").strip() or None
    touched: list[str] = []
    seen: set[str] = set()
    for fact_id in (winning_fact_id, *list(losing_fact_ids or [])):
        if fact_id and fact_id not in seen:
            seen.add(fact_id)
            touched.append(fact_id)
    prior_states = {
        fact_id: reader(fact_id, graph=graph, brain_id=brain_id) for fact_id in touched
    }

    def _call(action: str, fact_id: str) -> VerificationApplyResult:
        return apply(
            graph,
            fact_id=fact_id,
            brain_id=brain_id,
            action=action,
            actor=principal,
            verified_by=principal,
            note=note,
        )

    def _rollback(mutated: list[str]) -> VerifiedSideApplyResult:
        errors: list[str] = []
        for fact_id in reversed(mutated):
            action = _restore_action_for_prior_state(prior_states.get(fact_id))
            restored = _call(action, fact_id)
            if restored != "applied":
                detail = f"{fact_id} {action} returned {restored}"
                errors.append(detail)
                logger.error(
                    "verified_side rollback failed brain_id=%s fact_id=%s action=%s result=%s",
                    brain_id,
                    fact_id,
                    action,
                    restored,
                )
        if errors:
            logger.error(
                "verified_side could not restore prior verification states brain_id=%s winner=%s errors=%s",
                brain_id,
                winning_fact_id,
                errors,
            )
        return VerifiedSideApplyResult(
            "failed",
            rollback_failed=bool(errors),
            rollback_errors=tuple(errors),
        )

    applied = _call("verify", winning_fact_id)
    if applied != "applied":
        return VerifiedSideApplyResult(applied)
    mutated = [winning_fact_id]
    for fact_id in losing_fact_ids:
        rejected = _call("reject", fact_id)
        if rejected != "applied":
            return _rollback(mutated)
        mutated.append(fact_id)
    return VerifiedSideApplyResult("applied")


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


def close_open_conflicts_for_verified_fact(
    adapter: Any | None,
    brain_id: str,
    fact_id: str,
    *,
    graph: Any | None = None,
    notes: str | None = None,
) -> list[Any]:
    """Close open conflicts after #113 verifies a fact. Errors are swallowed."""
    if adapter is None or not str(fact_id or "").strip():
        return []
    try:
        from src.core.saving.conflicts import resolve_by_verified_fact

        return list(
            resolve_by_verified_fact(
                adapter,
                brain_id,
                str(fact_id).strip(),
                notes=notes,
                graph=graph,
            )
            or []
        )
    except Exception:
        return []
