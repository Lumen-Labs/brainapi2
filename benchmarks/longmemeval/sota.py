from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Any


_NORMALIZE_RE = re.compile(r"[^a-z0-9\s]+")
_WS_RE = re.compile(r"\s+")


def normalize_answer(text: str) -> str:
    lowered = (text or "").strip().lower()
    lowered = _NORMALIZE_RE.sub(" ", lowered)
    return _WS_RE.sub(" ", lowered).strip()


def majority_vote(answers: list[str]) -> str:
    nonempty = [a.strip() for a in answers if (a or "").strip()]
    if not nonempty:
        return ""
    counts = Counter(normalize_answer(a) for a in nonempty)
    winner_norm, _ = max(
        counts.items(),
        key=lambda item: (
            item[1],
            -next(i for i, a in enumerate(nonempty) if normalize_answer(a) == item[0]),
        ),
    )
    for answer in nonempty:
        if normalize_answer(answer) == winner_norm:
            return answer
    return nonempty[0]


@dataclass(frozen=True)
class GapFillPlan:
    needs_retry: bool
    reformulated_query: str
    missing_slots: list[str]


def plan_gap_fill(
    question: str,
    draft_answer: str,
    context: dict[str, Any],
) -> GapFillPlan:
    draft = (draft_answer or "").strip()
    text_blob = " ".join(
        [
            str(context.get("text_context") or ""),
            " ".join(str(p) for p in (context.get("source_passages") or [])[:20]),
            " ".join(str(t) for t in (context.get("triples") or [])[:40]),
        ]
    ).lower()
    abstained = bool(
        re.search(
            r"not mentioned|no information|incomplete|unknown|cannot (tell|determine)|"
            r"not available|do not have",
            draft,
            re.I,
        )
    )
    q_tokens = [t for t in re.findall(r"[a-z0-9]{3,}", question.lower()) if t]
    covered = sum(1 for t in q_tokens if t in text_blob) if q_tokens else 0
    coverage = covered / len(q_tokens) if q_tokens else 1.0
    missing = [t for t in q_tokens if t not in text_blob][:8]
    needs = abstained or (coverage < 0.45 and len(draft) < 40)
    reform = question.strip()
    if missing:
        reform = f"{question.strip()} (focus: {', '.join(missing[:5])})"
    return GapFillPlan(
        needs_retry=needs,
        reformulated_query=reform,
        missing_slots=missing,
    )


def merge_contexts(
    primary: dict[str, Any], secondary: dict[str, Any]
) -> dict[str, Any]:
    out = dict(primary)
    for key in (
        "source_passages",
        "historical_context",
        "triples",
        "paths",
        "degraded",
    ):
        a = list(primary.get(key) or [])
        b = list(secondary.get(key) or [])
        seen: set[str] = set()
        merged: list[Any] = []
        for item in a + b:
            sig = str(item)
            if sig in seen:
                continue
            seen.add(sig)
            merged.append(item)
        out[key] = merged
    text_a = str(primary.get("text_context") or "")
    text_b = str(secondary.get("text_context") or "")
    if text_b and text_b not in text_a:
        out["text_context"] = (text_a + "\n" + text_b).strip()
    return out
