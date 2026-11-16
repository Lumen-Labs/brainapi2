from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from evals.baseline.bm25 import BM25Index
from evals.baseline.chunk import chunk_corpus
from evals.baseline.dates import filter_as_of
from evals.baseline.dense import Embedder, cosine, make_embedder


def numeric_cost(cost: dict[str, Any] | None) -> dict[str, int]:
    out: dict[str, int] = {}
    for key, value in (cost or {}).items():
        if isinstance(value, bool) or isinstance(value, str):
            continue
        try:
            out[str(key)] = int(value)
        except (TypeError, ValueError):
            continue
    return out
from evals.baseline.fusion import fuse
from evals.baseline.rerank import CrossEncoderReranker, rerank_candidates

DEFAULT_HYBRID_CONFIG: dict[str, Any] = {
    "chunk_size": 800,
    "chunk_overlap": 120,
    "bm25_k1": 1.5,
    "bm25_b": 0.75,
    "fusion": "rrf",
    "rrf_k": 60,
    "dense_weight": 0.5,
    "retrieve_k": 30,
    "rerank_depth": 15,
    "reranker": "cross-encoder",
    "dense": "sentence-transformers",
    "dense_model": "sentence-transformers/all-MiniLM-L6-v2",
    "dense_dim": 256,
    "conflict_threshold": 0.5,
}

SMOKE_HYBRID_CONFIG: dict[str, Any] = {
    **DEFAULT_HYBRID_CONFIG,
    "reranker": "lexical",
    "dense": "hash",
    "dense_dim": 256,
}

HASH_INDEX_CONFIG: dict[str, Any] = {
    "reranker": "lexical",
    "dense": "hash",
    "dense_dim": 256,
}


@dataclass
class HybridIndex:
    config: dict[str, Any]
    chunks: list[dict[str, Any]]
    bm25: BM25Index
    vectors: list[list[float]]
    embedder: Embedder
    cross_encoder: CrossEncoderReranker | None
    build_ms: float
    cost: dict[str, int] = field(default_factory=dict)
    by_id: dict[str, dict[str, Any]] = field(default_factory=dict)

    def search(
        self,
        query: str,
        *,
        k: int | None = None,
        as_of: str | None = None,
    ) -> list[dict[str, Any]]:
        allowed = {chunk["id"] for chunk in filter_as_of(self.chunks, as_of)}
        bm25_hits = [
            (doc_id, score)
            for doc_id, score in self.bm25.search(query, k=max(50, int(self.config["retrieve_k"])))
            if doc_id in allowed
        ]
        q_vec, embed_cost = self.embedder.embed([query])
        for key, value in numeric_cost(embed_cost).items():
            self.cost[key] = self.cost.get(key, 0) + value
        dense_hits: list[tuple[str, float]] = []
        if q_vec:
            scored = []
            for chunk, vec in zip(self.chunks, self.vectors):
                if chunk["id"] not in allowed:
                    continue
                scored.append((chunk["id"], cosine(q_vec[0], vec)))
            scored.sort(key=lambda item: item[1], reverse=True)
            dense_hits = scored[: max(50, int(self.config["retrieve_k"]))]
        fused = fuse(
            bm25_hits,
            dense_hits,
            method=str(self.config.get("fusion") or "rrf"),
            rrf_k=int(self.config.get("rrf_k") or 60),
            dense_weight=float(self.config.get("dense_weight") or 0.5),
        )
        fused_ids = [item for item, _ in fused]
        for doc_id, _ in bm25_hits + dense_hits:
            if doc_id not in fused_ids:
                fused_ids.append(doc_id)
        candidates = []
        for doc_id in fused_ids:
            chunk = self.by_id.get(doc_id)
            if chunk:
                candidates.append(dict(chunk))
        ranked = rerank_candidates(
            query,
            candidates,
            method=str(self.config.get("reranker") or "lexical"),
            depth=int(self.config.get("rerank_depth") or 15),
            cross_encoder=self.cross_encoder,
        )
        limit = int(k or self.config.get("retrieve_k") or 10)
        return ranked[:limit]

    def with_config(self, config: dict[str, Any]) -> "HybridIndex":
        return HybridIndex(
            config=dict(config),
            chunks=self.chunks,
            bm25=self.bm25,
            vectors=self.vectors,
            embedder=self.embedder,
            cross_encoder=self.cross_encoder,
            build_ms=self.build_ms,
            cost=dict(self.cost),
            by_id=self.by_id,
        )


def build_index(
    docs: list[dict[str, Any]],
    config: dict[str, Any] | None,
    *,
    embedder: Embedder | None = None,
    cross_encoder: CrossEncoderReranker | None = None,
) -> HybridIndex:
    cfg = dict(DEFAULT_HYBRID_CONFIG)
    if config:
        cfg.update(config)
    started = time.perf_counter()
    chunks = chunk_corpus(
        docs,
        chunk_size=int(cfg["chunk_size"]),
        chunk_overlap=int(cfg["chunk_overlap"]),
    )
    bm25 = BM25Index(k1=float(cfg["bm25_k1"]), b=float(cfg["bm25_b"]))
    for chunk in chunks:
        bm25.add(chunk["id"], f"{chunk.get('title') or ''} {chunk.get('text') or ''}")
    embedder = embedder or make_embedder(
        str(cfg.get("dense") or "sentence-transformers"),
        model=str(cfg.get("dense_model") or ""),
        dim=int(cfg.get("dense_dim") or 256),
    )
    texts = [f"{chunk.get('title') or ''} {chunk.get('text') or ''}" for chunk in chunks]
    vectors, cost = embedder.embed(texts) if texts else ([], {})
    if len(vectors) < len(chunks):
        vectors.extend([[0.0] * int(cfg.get("dense_dim") or 256)] * (len(chunks) - len(vectors)))
    index = HybridIndex(
        config=cfg,
        chunks=chunks,
        bm25=bm25,
        vectors=vectors,
        embedder=embedder,
        cross_encoder=cross_encoder,
        build_ms=(time.perf_counter() - started) * 1000,
        cost=numeric_cost(cost),
        by_id={chunk["id"]: chunk for chunk in chunks},
    )
    return index
