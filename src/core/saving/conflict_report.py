"""JSON and markdown conflict reports for REST and MCP."""

from __future__ import annotations

from collections import Counter
from typing import Any

from src.constants.conflicts import FactConflict
from src.core.saving.conflicts import list_conflicts
from src.core.saving.validity import format_instant, utc_now


def _side_line(side) -> str:
    source = side.source.source_key or side.source.document or "unknown source"
    window = side.valid_from or "open"
    if side.valid_to:
        window = f"{window} .. {side.valid_to}"
    else:
        window = f"{window} .. now"
    return (
        f"- `{side.value}` ({side.polarity}"
        + (f", amount={side.amount}" if side.amount is not None else "")
        + f") from `{source}` valid {window}"
    )


def render_markdown(conflicts: list[FactConflict], summary: dict[str, Any]) -> str:
    lines = [
        "# Conflict report",
        "",
        f"Generated {summary.get('generated_at')}",
        "",
        (
            f"Open: {summary.get('open', 0)} | "
            f"Resolved: {summary.get('resolved', 0)} | "
            f"Dismissed: {summary.get('dismissed', 0)} | "
            f"Total: {summary.get('total', 0)}"
        ),
        "",
    ]
    by_type = summary.get("by_type") or {}
    if by_type:
        lines.append("By disagreement type:")
        for kind, count in by_type.items():
            lines.append(f"- {kind}: {count}")
        lines.append("")
    if not conflicts:
        lines.append("No conflicts matched the filters.")
        return "\n".join(lines)
    grouped: dict[str, list[FactConflict]] = {}
    for conflict in conflicts:
        grouped.setdefault(conflict.status, []).append(conflict)
    for status in ("open", "resolved", "dismissed"):
        rows = grouped.get(status) or []
        if not rows:
            continue
        lines.append(f"## {status.capitalize()}")
        lines.append("")
        for conflict in rows:
            subject = conflict.subject.name or conflict.subject.uuid
            lines.append(f"### {subject} `{conflict.predicate}`")
            lines.append(
                f"- id: `{conflict.id}` | type: {conflict.disagreement_type} | "
                f"confidence: {conflict.confidence:.2f} | method: {conflict.detection_method}"
            )
            for side in conflict.sides:
                lines.append(_side_line(side))
            if conflict.resolution and conflict.resolution.method:
                winner = conflict.resolution.winning_fact_id or ""
                extra = f" winning=`{winner}`" if winner else ""
                lines.append(
                    f"- resolution: {conflict.resolution.method}{extra}"
                    + (f" — {conflict.resolution.notes}" if conflict.resolution.notes else "")
                )
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def build_report(
    adapter: Any,
    brain_id: str,
    *,
    status: str | None = None,
    source: str | None = None,
    subject: str | None = None,
    since: Any = None,
    until: Any = None,
    limit: int = 200,
    skip: int = 0,
    fmt: str = "json",
) -> dict[str, Any]:
    conflicts, total = list_conflicts(
        adapter,
        brain_id,
        status=status,
        source=source,
        subject=subject,
        since=since,
        until=until,
        limit=limit,
        skip=skip,
    )
    counts = Counter(item.status for item in conflicts)
    types = Counter(item.disagreement_type for item in conflicts)
    summary = {
        "generated_at": format_instant(utc_now()),
        "total": total,
        "returned": len(conflicts),
        "open": counts.get("open", 0),
        "resolved": counts.get("resolved", 0),
        "dismissed": counts.get("dismissed", 0),
        "by_type": dict(types),
        "status": status,
        "source": source,
        "subject": subject,
        "since": since,
        "until": until,
    }
    payload: dict[str, Any] = {
        "summary": summary,
        "conflicts": [item.model_dump(mode="json") for item in conflicts],
    }
    if fmt == "markdown":
        payload["markdown"] = render_markdown(conflicts, summary)
    return payload
