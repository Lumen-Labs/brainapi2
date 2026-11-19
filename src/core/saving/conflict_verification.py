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
class VerificationSnapshot:
    state: str
    verified_by: str | None = None
    verified_at: str | None = None
    verification_source: str | None = None
    reason: str | None = None
    kind: str | None = None


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


def read_fact_verification_snapshot(
    fact_id: str,
    *,
    graph: Any | None = None,
    brain_id: str | None = None,
) -> VerificationSnapshot | None:
    """Return the full #113 verification record for a fact.

    Unlike ``read_fact_verification_state``, this does not swallow errors.
    ``None`` means the graph is absent or the fact has no record.
    """
    if graph is None or not brain_id or not fact_id:
        return None
    from src.core.verification.service import get_fact_record

    record = get_fact_record(graph, fact_id, brain_id)
    verification = getattr(record, "verification", None)
    if verification is None:
        return None
    state = str(getattr(verification, "state", "") or "").strip().lower()
    if not state:
        return None
    return VerificationSnapshot(
        state=state,
        verified_by=getattr(verification, "verified_by", None),
        verified_at=getattr(verification, "verified_at", None),
        verification_source=getattr(verification, "verification_source", None),
        reason=getattr(verification, "note", None),
        kind=getattr(record, "kind", None),
    )


def restore_fact_verification_snapshot(
    graph: Any | None,
    *,
    fact_id: str,
    brain_id: str,
    snapshot: VerificationSnapshot,
    from_state: str | None = None,
    actor: str | None = None,
) -> VerificationApplyResult:
    """Write a snapshot back through ``graph.set_fact_verification``.

    Adds one history entry tagged ``rollback`` and restores the original
    verifier, timestamp, source, and reason instead of re-applying verify/reject/reset.
    """
    if graph is None:
        return "absent"
    setter = getattr(graph, "set_fact_verification", None)
    if not callable(setter):
        return "failed"
    try:
        from src.core.verification.facts import (
            UNVERIFIED,
            FactHistoryEntry,
            normalize_state,
            utc_now_iso,
            verification_write_properties,
        )
    except ImportError:
        return "absent"
    target = normalize_state(snapshot.state)
    unverified = target == UNVERIFIED
    properties = verification_write_properties(
        state=target,
        verified_by=None if unverified else snapshot.verified_by,
        verified_at=None if unverified else snapshot.verified_at,
        verification_source=None if unverified else snapshot.verification_source,
        note=None if unverified else snapshot.reason,
        history=[],
    )
    properties.pop("verification_history", None)
    history_entry = FactHistoryEntry(
        from_state=normalize_state(from_state),
        to_state=target,
        verified_by=(actor or "").strip() or None,
        verified_at=utc_now_iso(),
        verification_source="rollback",
        note="rollback",
    )
    try:
        updated = setter(
            fact_id,
            brain_id,
            kind=snapshot.kind or "relationship",
            properties=properties,
            history_entry=history_entry,
        )
    except Exception:
        return "failed"
    if updated is None:
        return "failed"
    return "applied"


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
    read_snapshot: Callable[..., VerificationSnapshot | None] | None = None,
    restore_snapshot: Callable[..., VerificationApplyResult] | None = None,
) -> VerifiedSideApplyResult:
    """Verify the winning fact and reject the other conflict side via #113.

    Snapshot full verification properties before writing. If a later write
    fails, restore those exact properties. If the graph is present and any
    pre-read is missing or errors, abort before writing. The caller must not
    mark the conflict resolved unless this returns ``applied``.
    """
    apply = apply_action or apply_fact_verification_action
    reader = read_snapshot or read_fact_verification_snapshot
    restorer = restore_snapshot or restore_fact_verification_snapshot
    principal = (actor or verified_by or "").strip() or None
    touched: list[str] = []
    seen: set[str] = set()
    for fact_id in (winning_fact_id, *list(losing_fact_ids or [])):
        if fact_id and fact_id not in seen:
            seen.add(fact_id)
            touched.append(fact_id)

    snapshots: dict[str, VerificationSnapshot] = {}
    if graph is not None:
        for fact_id in touched:
            try:
                snapshot = reader(fact_id, graph=graph, brain_id=brain_id)
            except Exception as exc:
                detail = f"{fact_id} pre-read error: {exc}"
                logger.error(
                    "verified_side pre-read failed brain_id=%s fact_id=%s error=%s",
                    brain_id,
                    fact_id,
                    exc,
                )
                return VerifiedSideApplyResult("failed", rollback_errors=(detail,))
            if snapshot is None:
                detail = f"{fact_id} pre-read returned none"
                logger.error(
                    "verified_side pre-read missing brain_id=%s fact_id=%s",
                    brain_id,
                    fact_id,
                )
                return VerifiedSideApplyResult("failed", rollback_errors=(detail,))
            snapshots[fact_id] = snapshot

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
            snapshot = snapshots.get(fact_id)
            if snapshot is None:
                detail = f"{fact_id} missing snapshot"
                errors.append(detail)
                logger.error(
                    "verified_side rollback missing snapshot brain_id=%s fact_id=%s",
                    brain_id,
                    fact_id,
                )
                continue
            from_state = "verified" if fact_id == winning_fact_id else "rejected"
            restored = restorer(
                graph,
                fact_id=fact_id,
                brain_id=brain_id,
                snapshot=snapshot,
                from_state=from_state,
                actor=principal,
            )
            if restored != "applied":
                detail = f"{fact_id} restore returned {restored}"
                errors.append(detail)
                logger.error(
                    "verified_side rollback failed brain_id=%s fact_id=%s result=%s",
                    brain_id,
                    fact_id,
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
