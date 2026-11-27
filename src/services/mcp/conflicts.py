from __future__ import annotations

from typing import Any

from src.core.saving.conflict_report import build_report
from src.core.auth.scopes import VERIFICATION_PERMISSION_DENIED
from src.core.saving.conflicts import (
    VerificationPermissionDenied,
    dismiss_conflict,
    get_conflict,
    list_conflicts,
    reopen_conflict,
    resolve_conflict_authorized,
)
from src.core.saving.validity import InvalidAsOf, optional_instant
from src.services.mcp.versions import clamp_page

CONFLICT_LIMIT = 50
REPORT_LIMIT = 200


def _dump(row: Any) -> Any:
    dump = getattr(row, "model_dump", None)
    if callable(dump):
        return dump(mode="json")
    return row


def read_conflicts(
    adapter: Any,
    brain_id: str,
    *,
    authorized: bool,
    status: str | None = None,
    source: str | None = None,
    subject: str | None = None,
    since: Any = None,
    until: Any = None,
    limit: Any = 10,
    skip: Any = 0,
    graph: Any | None = None,
) -> dict[str, Any] | str:
    if not authorized:
        return "Unauthorized"
    bounded = clamp_page(limit, 10, CONFLICT_LIMIT)
    try:
        offset = max(0, int(skip or 0))
    except (TypeError, ValueError):
        offset = 0
    try:
        since_dt = optional_instant(since, name="since")
        until_dt = optional_instant(until, name="until")
    except InvalidAsOf as exc:
        return f"Error: {exc}"
    if status and status not in {"open", "resolved", "dismissed"}:
        return "Error: status must be open, resolved, or dismissed"
    rows, total = list_conflicts(
        adapter,
        brain_id,
        status=status or None,
        source=source or None,
        subject=subject or None,
        since=since_dt,
        until=until_dt,
        limit=bounded,
        skip=offset,
        graph=graph,
    )
    return {
        "conflicts": [_dump(row) for row in rows],
        "count": len(rows),
        "total": total,
    }


def read_conflict(
    adapter: Any,
    brain_id: str,
    conflict_id: str,
    *,
    authorized: bool,
    graph: Any | None = None,
) -> dict[str, Any] | str:
    if not authorized:
        return "Unauthorized"
    if not str(conflict_id or "").strip():
        return "Error: conflict id is required"
    conflict = get_conflict(
        adapter, brain_id, str(conflict_id).strip(), graph=graph
    )
    if conflict is None:
        return "Error: conflict not found"
    return {"conflict": _dump(conflict)}


def resolve_conflict_tool(
    adapter: Any,
    brain_id: str,
    conflict_id: str,
    *,
    authorized: bool,
    winning_fact_id: str | None = None,
    notes: str | None = None,
    method: str = "manual",
    graph: Any | None = None,
    actor: str | None = None,
    has_verification: bool = False,
) -> dict[str, Any] | str:
    if not str(conflict_id or "").strip():
        return "Error: conflict id is required"
    try:
        conflict = resolve_conflict_authorized(
            adapter,
            brain_id,
            str(conflict_id).strip(),
            authorized=authorized,
            actor=actor,
            has_verification=has_verification,
            winning_fact_id=winning_fact_id or None,
            notes=notes or None,
            method=method or "manual",
            graph=graph,
        )
    except VerificationPermissionDenied:
        return (
            f"Error: {VERIFICATION_PERMISSION_DENIED}: "
            "verification scope is required"
        )
    except PermissionError:
        return "Unauthorized"
    except KeyError:
        return "Error: conflict not found"
    except ValueError as exc:
        return f"Error: {exc}"
    return {"conflict": _dump(conflict)}


def dismiss_conflict_tool(
    adapter: Any,
    brain_id: str,
    conflict_id: str,
    *,
    authorized: bool,
    notes: str | None = None,
    actor: str | None = None,
    has_verification: bool = False,
) -> dict[str, Any] | str:
    if not authorized:
        return "Unauthorized"
    if not has_verification or not (actor or "").strip():
        return (
            f"Error: {VERIFICATION_PERMISSION_DENIED}: "
            "verification scope is required"
        )
    if not str(conflict_id or "").strip():
        return "Error: conflict id is required"
    try:
        conflict = dismiss_conflict(
            adapter,
            brain_id,
            str(conflict_id).strip(),
            notes=notes or None,
            resolved_by=(actor or "").strip() or "system",
        )
    except KeyError:
        return "Error: conflict not found"
    return {"conflict": _dump(conflict)}


def reopen_conflict_tool(
    adapter: Any,
    brain_id: str,
    conflict_id: str,
    *,
    authorized: bool,
    notes: str | None = None,
    actor: str | None = None,
    graph: Any | None = None,
    has_verification: bool = False,
) -> dict[str, Any] | str:
    if not authorized:
        return "Unauthorized"
    if not has_verification or not (actor or "").strip():
        return (
            f"Error: {VERIFICATION_PERMISSION_DENIED}: "
            "verification scope is required"
        )
    if not str(conflict_id or "").strip():
        return "Error: conflict id is required"
    try:
        conflict = reopen_conflict(
            adapter,
            brain_id,
            str(conflict_id).strip(),
            notes=notes or None,
            resolved_by=(actor or "").strip() or "system",
            graph=graph,
        )
    except KeyError:
        return "Error: conflict not found"
    except ValueError as exc:
        return f"Error: {exc}"
    return {"conflict": _dump(conflict)}


def conflict_report_tool(
    adapter: Any,
    brain_id: str,
    *,
    authorized: bool,
    status: str | None = None,
    source: str | None = None,
    subject: str | None = None,
    since: Any = None,
    until: Any = None,
    fmt: str = "json",
    limit: Any = 200,
) -> dict[str, Any] | str:
    if not authorized:
        return "Unauthorized"
    if fmt not in {"json", "markdown"}:
        return "Error: format must be json or markdown"
    bounded = clamp_page(limit, 200, REPORT_LIMIT)
    try:
        since_dt = optional_instant(since, name="since")
        until_dt = optional_instant(until, name="until")
    except InvalidAsOf as exc:
        return f"Error: {exc}"
    return build_report(
        adapter,
        brain_id,
        status=status or None,
        source=source or None,
        subject=subject or None,
        since=since_dt,
        until=until_dt,
        limit=bounded,
        fmt=fmt,
    )
