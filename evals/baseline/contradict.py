from __future__ import annotations

import re
from typing import Any

from evals.baseline.index import HybridIndex
from evals.baseline.textlib import content_tokens, tokenize
from evals.metrics import pair_key

NUMBER_RE = re.compile(r"^\d+(?:\.\d+)?$")
NEGATIONS = frozenset({"not", "never", "no", "cannot", "can't", "don't", "optional", "disabled"})
REQUIRED = frozenset({"required", "must", "mandatory", "always", "enforced"})
DURATION = {
    "day": "day",
    "days": "day",
    "hour": "hour",
    "hours": "hour",
    "week": "week",
    "weeks": "week",
    "month": "month",
    "months": "month",
    "year": "year",
    "years": "year",
    "character": "character",
    "characters": "character",
    "minute": "minute",
    "minutes": "minute",
}
PLAN_IN_TITLE = re.compile(
    r"\((standard|enterprise|free|paid)\s+plan\)", re.IGNORECASE
)
GENERIC_TOPIC = frozenset(
    {
        "days",
        "day",
        "hours",
        "hour",
        "within",
        "after",
        "before",
        "help",
        "cloud",
        "northwind",
        "workspace",
        "account",
        "submit",
        "article",
        "notice",
        "getting",
        "started",
        "create",
        "strong",
        "only",
        "later",
        "profile",
        "security",
        "whitepaper",
        "lifecycle",
        "what",
        "happens",
        "admin",
        "console",
        "legal",
        "terms",
        "snippet",
        "billing",
        "policy",
        "required",
        "optional",
        "standard",
        "enterprise",
        "plan",
        "faq",
    }
)
SYN = {
    "2fa": "twofactor",
    "twofactor": "twofactor",
    "two": "two",
    "factor": "factor",
    "authentication": "twofactor",
    "refund": "refund",
    "refunds": "refund",
    "password": "password",
    "passwords": "password",
    "retention": "retention",
    "retained": "retention",
    "delete": "retention",
    "deleted": "retention",
}


def _norm_tokens(text: str) -> list[str]:
    toks = content_tokens(text)
    out: list[str] = []
    i = 0
    while i < len(toks):
        if i + 1 < len(toks) and toks[i] == "two" and toks[i + 1] == "factor":
            out.append("twofactor")
            i += 2
            continue
        out.append(SYN.get(toks[i], toks[i]))
        i += 1
    return out


def _quantities(text: str) -> dict[str, set[str]]:
    toks = tokenize(text)
    out: dict[str, set[str]] = {}
    for i, tok in enumerate(toks):
        if not NUMBER_RE.fullmatch(tok):
            continue
        unit = "count"
        if i + 1 < len(toks):
            nxt = toks[i + 1]
            if nxt in DURATION:
                unit = DURATION[nxt]
        out.setdefault(unit, set()).add(tok)
    return out


def _number_clash(a: str, b: str) -> bool:
    qa, qb = _quantities(a), _quantities(b)
    for unit in set(qa) & set(qb):
        if unit == "count":
            continue
        if qa[unit] != qb[unit]:
            return True
    return False


def _polarity_clash(ta: set[str], tb: set[str]) -> bool:
    left = {"optional", "disabled"}
    right = {"required", "mandatory", "enforced", "always"}
    if (ta & left and tb & right) or (tb & left and ta & right):
        return True
    return False


def _title_plan(doc: dict[str, Any]) -> str | None:
    title = str(doc.get("title") or "")
    match = PLAN_IN_TITLE.search(title)
    if match:
        return match.group(1).lower()
    return None


def _topic_tokens(doc: dict[str, Any]) -> set[str]:
    title = _norm_tokens(str(doc.get("title") or ""))
    return {
        tok
        for tok in title
        if len(tok) > 3
        and not NUMBER_RE.fullmatch(tok)
        and tok not in GENERIC_TOPIC
    }


def _exclusive_audiences(left: dict[str, Any], right: dict[str, Any], a: str, b: str) -> bool:
    plan_l, plan_r = _title_plan(left), _title_plan(right)
    if plan_l and plan_r and plan_l != plan_r:
        return True
    lowered = (a.lower(), b.lower())
    if any("not the current" in text or "sunset" in text for text in lowered):
        return True
    return False


def contradiction_score(left: dict[str, Any], right: dict[str, Any]) -> float:
    if (left.get("statement_id") or left.get("doc_id") or left.get("id")) == (
        right.get("statement_id") or right.get("doc_id") or right.get("id")
    ):
        return 0.0
    a = f"{left.get('title') or ''} {left.get('text') or ''}"
    b = f"{right.get('title') or ''} {right.get('text') or ''}"
    if _exclusive_audiences(left, right, a, b):
        return 0.0
    ta = set(_norm_tokens(a))
    tb = set(_norm_tokens(b))
    if not ta or not tb:
        return 0.0
    overlap = len(ta & tb) / max(1, min(len(ta), len(tb)))
    topics = _topic_tokens(left) & _topic_tokens(right)
    if not topics:
        return 0.0
    number = _number_clash(a, b)
    polarity = _polarity_clash(ta, tb)
    if not number and not polarity:
        return 0.0
    score = 0.0
    if number:
        score += 0.6
    if polarity:
        score += 0.5
    score += 0.1 * min(1.0, overlap)
    return min(1.0, score)


def detect_conflicts(
    index: HybridIndex,
    docs: list[dict[str, Any]],
    *,
            threshold: float = 0.5,
    neighbors: int = 8,
) -> list[dict[str, Any]]:
    seen: set[tuple[str, str]] = set()
    pairs: list[dict[str, Any]] = []
    by_id = {str(doc.get("id")): doc for doc in docs}
    for doc in docs:
        query = f"{doc.get('title') or ''} {doc.get('text') or ''}"
        hits = index.search(query, k=neighbors)
        left_id = str(doc.get("statement_id") or doc.get("id"))
        for hit in hits:
            right_doc = by_id.get(str(hit.get("doc_id") or ""))
            if right_doc is None:
                right_doc = hit
            right_id = str(
                right_doc.get("statement_id")
                or hit.get("statement_id")
                or hit.get("doc_id")
                or hit.get("id")
            )
            if not left_id or not right_id or left_id == right_id:
                continue
            key = pair_key(left_id, right_id)
            if key in seen:
                continue
            score = contradiction_score(doc, right_doc)
            if score < threshold:
                continue
            seen.add(key)
            pairs.append(
                {
                    "left": key[0],
                    "right": key[1],
                    "score": score,
                    "sources": [left_id, right_id],
                }
            )
    pairs.sort(key=lambda row: -float(row["score"]))
    return pairs
