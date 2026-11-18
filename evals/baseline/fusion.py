from __future__ import annotations

from collections import defaultdict


def reciprocal_rank_fusion(
    ranked_lists: list[list[str]],
    *,
    k: int = 60,
) -> list[tuple[str, float]]:
    scores: dict[str, float] = defaultdict(float)
    for ranked in ranked_lists:
        for rank, item in enumerate(ranked):
            if not item:
                continue
            scores[item] += 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda pair: pair[1], reverse=True)


def weighted_fusion(
    bm25: list[tuple[str, float]],
    dense: list[tuple[str, float]],
    *,
    dense_weight: float = 0.5,
) -> list[tuple[str, float]]:
    def norm(pairs: list[tuple[str, float]]) -> dict[str, float]:
        if not pairs:
            return {}
        vals = [max(0.0, score) for _, score in pairs]
        lo = min(vals)
        hi = max(vals)
        span = (hi - lo) or 1.0
        return {item: (max(0.0, score) - lo) / span for item, score in pairs}

    bm25_n = norm(bm25)
    dense_n = norm(dense)
    alpha = min(1.0, max(0.0, float(dense_weight)))
    keys = set(bm25_n) | set(dense_n)
    fused = [
        (key, (1.0 - alpha) * bm25_n.get(key, 0.0) + alpha * dense_n.get(key, 0.0))
        for key in keys
    ]
    return sorted(fused, key=lambda pair: pair[1], reverse=True)


def fuse(
    bm25: list[tuple[str, float]],
    dense: list[tuple[str, float]],
    *,
    method: str = "rrf",
    rrf_k: int = 60,
    dense_weight: float = 0.5,
) -> list[tuple[str, float]]:
    name = (method or "rrf").strip().lower()
    if name in {"weighted", "linear", "cc"}:
        return weighted_fusion(bm25, dense, dense_weight=dense_weight)
    return reciprocal_rank_fusion(
        [[item for item, _ in bm25], [item for item, _ in dense]],
        k=rrf_k,
    )
