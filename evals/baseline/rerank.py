from __future__ import annotations

from typing import Any

from evals.baseline.textlib import content_tokens, tokenize


def lexical_score(query: str, text: str) -> float:
    q = content_tokens(query)
    d = tokenize(text)
    if not q or not d:
        return 0.0
    d_set = set(d)
    hits = sum(1 for tok in q if tok in d_set)
    coverage = hits / len(q)
    overlap = hits / max(1, len(set(q) | d_set))
    return 0.7 * coverage + 0.3 * overlap


def lexical_rerank(
    query: str,
    candidates: list[dict[str, Any]],
    *,
    depth: int,
) -> list[dict[str, Any]]:
    head = candidates[: max(1, int(depth))]
    scored = []
    for item in head:
        score = lexical_score(query, f"{item.get('title') or ''} {item.get('text') or ''}")
        clone = dict(item)
        clone["rerank_score"] = score
        scored.append(clone)
    scored.sort(key=lambda row: (-float(row["rerank_score"]), str(row.get("id") or "")))
    return scored + [dict(item) for item in candidates[len(head) :]]


class CrossEncoderReranker:
    def __init__(self, model_name: str) -> None:
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name)
        self.name = "cross-encoder"

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        *,
        depth: int,
    ) -> list[dict[str, Any]]:
        head = candidates[: max(1, int(depth))]
        pairs = [
            [query, f"{item.get('title') or ''} {item.get('text') or ''}".strip()]
            for item in head
        ]
        scores = self.model.predict(pairs)
        ranked = []
        for item, score in zip(head, scores):
            clone = dict(item)
            clone["rerank_score"] = float(score)
            ranked.append(clone)
        ranked.sort(key=lambda row: (-float(row["rerank_score"]), str(row.get("id") or "")))
        return ranked + [dict(item) for item in candidates[len(head) :]]


def rerank_candidates(
    query: str,
    candidates: list[dict[str, Any]],
    *,
    method: str,
    depth: int,
    cross_encoder: CrossEncoderReranker | None = None,
) -> list[dict[str, Any]]:
    name = (method or "lexical").strip().lower()
    if name in {"none", "off"}:
        return candidates
    if name in {"cross-encoder", "ce", "cross_encoder"} and cross_encoder is not None:
        return cross_encoder.rerank(query, candidates, depth=depth)
    return lexical_rerank(query, candidates, depth=depth)
