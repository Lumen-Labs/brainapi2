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
        self.model_name = model_name
        self.name = "cross-encoder"
        self._predict = self._load(model_name)

    @staticmethod
    def _load(model_name: str):
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for the cross-encoder reranker. "
                "Install with `pip install -r evals/requirements-strong.txt`, "
                "or set EVAL_RERANKER=lexical. CI smoke uses lexical rerank."
            ) from exc
        try:
            model = CrossEncoder(model_name, trust_remote_code=True)

            def _predict(pairs: list[list[str]]):
                try:
                    return model.predict(pairs, show_progress_bar=False)
                except TypeError:
                    return model.predict(pairs)

            return _predict
        except Exception as exc:
            try:
                from FlagEmbedding import FlagReranker
            except ImportError as flag_exc:
                raise SystemExit(
                    f"Could not load reranker {model_name!r} via sentence-transformers "
                    f"({exc}). Install FlagEmbedding or pick a CrossEncoder-compatible "
                    "model such as BAAI/bge-reranker-large."
                ) from flag_exc
            flag = FlagReranker(model_name, use_fp16=False)
            return lambda pairs: flag.compute_score(pairs, normalize=True)

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        *,
        depth: int,
    ) -> list[dict[str, Any]]:
        head = candidates[: max(1, int(depth))]
        if not head:
            return candidates
        pairs = [
            [query, f"{item.get('title') or ''} {item.get('text') or ''}".strip()]
            for item in head
        ]
        raw = self._predict(pairs)
        if isinstance(raw, (int, float)):
            scores = [float(raw)]
        else:
            scores = [float(item) for item in list(raw)]
        ranked = []
        for item, score in zip(head, scores):
            clone = dict(item)
            clone["rerank_score"] = score
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
