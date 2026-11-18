from __future__ import annotations

from typing import Iterable

from src.core.intent.models import CheckResult, IntentFact, ResponseFormat

_MAX_OUTPUT_CHARS = 18000
_CONCISE_LIMIT = 8
_DETAILED_LIMIT = 20
_PAGE_CAP = 50


def clamp_page(limit: int | None, response_format: ResponseFormat) -> int:
    default = _CONCISE_LIMIT if response_format == "concise" else _DETAILED_LIMIT
    try:
        parsed = int(limit) if limit is not None else default
    except (TypeError, ValueError):
        parsed = default
    return max(1, min(parsed, _PAGE_CAP))


def _status_note(fact: IntentFact) -> str:
    parts = [fact.status]
    if fact.retired_reason:
        parts.append(f"retired because: {fact.retired_reason}")
    if fact.superseded_by:
        parts.append(f"superseded_by {fact.superseded_by}")
    if fact.contradicted_by:
        parts.append("contradicted_by " + ", ".join(fact.contradicted_by))
    return "; ".join(parts)


def _source_line(fact: IntentFact) -> str:
    source = fact.source
    bits = []
    if source.doc:
        bits.append(f"doc={source.doc}")
    if source.url:
        bits.append(f"url={source.url}")
    if source.source_key:
        bits.append(f"source_key={source.source_key}")
    writer = fact.writer or source.writer
    if writer:
        bits.append(f"writer={writer}")
    return ", ".join(bits)


def render_facts(
    facts: Iterable[IntentFact],
    *,
    response_format: ResponseFormat = "concise",
    heading: str | None = None,
) -> str:
    lines: list[str] = []
    if heading:
        lines.append(heading)
    items = list(facts)
    if not items:
        lines.append("No matching facts.")
        return "\n".join(lines)
    for fact in items:
        if response_format == "concise":
            lines.append(f"- {fact.natural_id()} ({_status_note(fact)})")
            continue
        lines.append(f"- {fact.natural_id()}")
        lines.append(f"  fact_id: {fact.fact_id}")
        lines.append(f"  status: {_status_note(fact)}")
        if fact.valid_at or fact.invalid_at:
            lines.append(f"  valid_at: {fact.valid_at or '-'}  invalid_at: {fact.invalid_at or '-'}")
        source_line = _source_line(fact)
        if source_line:
            lines.append(f"  source: {source_line}")
        if fact.text and fact.text != fact.natural_id():
            lines.append(f"  evidence: {fact.text}")
    text = "\n".join(lines)
    if len(text) <= _MAX_OUTPUT_CHARS:
        return text
    return text[: _MAX_OUTPUT_CHARS - 16].rstrip() + "\n… truncated"


def render_check(result: CheckResult, *, response_format: ResponseFormat = "concise") -> str:
    heading = f"{result.verdict}: {result.claim}"
    if result.as_of:
        heading = f"{heading} (as of {result.as_of})"
    return render_facts(result.evidence, response_format=response_format, heading=heading)


def estimate_tokens(text: str) -> int:
    try:
        import tiktoken

        return len(tiktoken.get_encoding("cl100k_base").encode(text))
    except Exception:
        return max(1, (len(text) + 3) // 4)
