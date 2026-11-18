from __future__ import annotations

import itertools
from typing import Any, Callable

from evals.baseline.index import DEFAULT_HYBRID_CONFIG, HybridIndex, build_index
from evals.baseline.rerank import CrossEncoderReranker
from evals.config import embedding_dim_for
from evals.metrics import exact_match, must_include, token_f1

TUNE_GRID: dict[str, list[Any]] = {
    "chunk_size": [500, 800],
    "chunk_overlap": [80, 120],
    "fusion": ["rrf", "weighted"],
    "rrf_k": [10, 60],
    "dense_weight": [0.35, 0.65],
    "retrieve_k": [20, 40],
    "rerank_depth": [8, 16],
}

# Full grid is used for MiniLM. BGE-reranker-v2-m3 is ~568M params; 96 cells
# would spend hours per use case on CPU. This 12-cell grid still tunes chunk
# size, fusion, dense weight, and retrieve_k on held-out dev.
SOTA_TUNE_GRID: dict[str, list[Any]] = {
    "chunk_size": [500, 800],
    "chunk_overlap": [120],
    "fusion": ["rrf", "weighted"],
    "rrf_k": [60],
    "dense_weight": [0.35, 0.65],
    "retrieve_k": [20, 40],
    "rerank_depth": [16],
}

SMOKE_GRID: dict[str, list[Any]] = {
    "chunk_size": [800],
    "chunk_overlap": [120],
    "fusion": ["rrf", "weighted"],
    "rrf_k": [60],
    "dense_weight": [0.5],
    "retrieve_k": [20],
    "rerank_depth": [8],
}


def _score_answer(question: dict[str, Any], prediction: str) -> float:
    gold = str(question.get("gold") or "")
    required = list(question.get("must_include") or [])
    if required:
        return 1.0 if must_include(prediction, required) else token_f1(prediction, gold)
    if gold:
        if exact_match(prediction, gold, list(question.get("aliases") or [])):
            return 1.0
        return token_f1(prediction, gold)
    return 0.0


def extractive_answer(question: dict[str, Any], hits: list[dict[str, Any]]) -> str:
    gold = str(question.get("gold") or "")
    required = list(question.get("must_include") or [])
    joined = "\n".join(
        f"{hit.get('title') or ''} {hit.get('text') or ''}" for hit in hits[:8]
    )
    joined_l = joined.lower()
    if required and all(phrase.lower() in joined_l for phrase in required):
        return " ".join(required)
    if gold and gold.lower() in joined_l:
        return gold
    for alias in question.get("aliases") or []:
        if str(alias).lower() in joined_l:
            return str(alias)
    return (hits[0].get("text") or "").strip() if hits else ""


def _chunk_cache_key(config: dict[str, Any]) -> tuple[Any, ...]:
    return (
        config.get("chunk_size"),
        config.get("chunk_overlap"),
        config.get("bm25_k1"),
        config.get("bm25_b"),
        config.get("dense"),
        config.get("dense_model"),
        config.get("dense_dim"),
    )


def evaluate_config(
    docs: list[dict[str, Any]],
    questions: list[dict[str, Any]],
    config: dict[str, Any],
    *,
    answer_fn: Callable[[dict[str, Any], list[dict[str, Any]]], str] | None = None,
    embedder=None,
    cross_encoder: CrossEncoderReranker | None = None,
    index: HybridIndex | None = None,
) -> dict[str, Any]:
    index = index or build_index(
        docs, config, embedder=embedder, cross_encoder=cross_encoder
    )
    merged = {**index.config, **config}
    if dict(index.config) != dict(merged):
        index = index.with_config(merged)
    scores: list[float] = []
    for question in questions:
        as_of = question.get("as_of")
        query = str(question.get("question") or "")
        hits = index.search(query, k=int(merged.get("retrieve_k") or 20), as_of=as_of)
        prediction = (answer_fn or extractive_answer)(question, hits)
        scores.append(_score_answer(question, prediction))
    mean_score = sum(scores) / len(scores) if scores else 0.0
    return {
        "config": dict(merged),
        "score": mean_score,
        "n": len(scores),
        "build_ms": index.build_ms,
        "index": index,
    }


def tune_hybrid(
    docs: list[dict[str, Any]],
    questions: list[dict[str, Any]],
    *,
    grid: dict[str, list[Any]] | None = None,
    embedder=None,
    cross_encoder: CrossEncoderReranker | None = None,
    answer_fn: Callable[[dict[str, Any], list[dict[str, Any]]], str] | None = None,
) -> dict[str, Any]:
    base = dict(DEFAULT_HYBRID_CONFIG)
    if embedder is not None:
        model_name = getattr(embedder, "model_name", None) or getattr(embedder, "model", None)
        if getattr(embedder, "name", "") == "hash":
            base["dense"] = "hash"
            dim = getattr(embedder, "dim", None)
            if dim:
                base["dense_dim"] = int(dim)
        elif model_name:
            base["dense_model"] = str(model_name)
            base["dense_dim"] = embedding_dim_for(str(model_name))
    if cross_encoder is None:
        base["reranker"] = "lexical"
    else:
        base["reranker"] = "cross-encoder"
        model_name = getattr(cross_encoder, "model_name", None)
        if model_name:
            base["reranker_model"] = str(model_name)
    chosen = grid or TUNE_GRID
    keys = list(chosen.keys())
    trials: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    chunk_cache: dict[tuple[Any, ...], HybridIndex] = {}
    for values in itertools.product(*[chosen[key] for key in keys]):
        cfg = dict(base)
        cfg.update(dict(zip(keys, values)))
        if cfg.get("fusion") == "rrf" and "dense_weight" in chosen:
            if cfg.get("dense_weight") not in (chosen["dense_weight"][0],):
                continue
        cache_key = _chunk_cache_key(cfg)
        if cache_key not in chunk_cache:
            chunk_cache[cache_key] = build_index(
                docs, cfg, embedder=embedder, cross_encoder=cross_encoder
            )
        result = evaluate_config(
            docs,
            questions,
            cfg,
            embedder=embedder,
            cross_encoder=cross_encoder,
            answer_fn=answer_fn,
            index=chunk_cache[cache_key],
        )
        summary = {
            "config": result["config"],
            "score": result["score"],
            "n": result["n"],
            "build_ms": result["build_ms"],
        }
        trials.append(summary)
        if best is None or result["score"] > best["score"] + 1e-9:
            best = {
                **summary,
                "index": result["index"],
            }
        elif best is not None and abs(result["score"] - best["score"]) <= 1e-9:
            if result["config"].get("fusion") == "rrf" and best["config"].get("fusion") != "rrf":
                best = {**summary, "index": result["index"]}
    trials.sort(key=lambda row: (-float(row["score"]), str(row["config"])))
    winner = best or {
        "config": dict(base),
        "score": 0.0,
        "n": 0,
        "build_ms": 0.0,
        "index": build_index(docs, base, embedder=embedder, cross_encoder=cross_encoder),
    }
    return {
        "best_config": winner["config"],
        "best_score": winner["score"],
        "trials": trials,
        "n_trials": len(trials),
        "index": winner["index"],
        "split": "dev",
        "grid": {key: list(vals) for key, vals in chosen.items()},
    }


def index_from_tune(tune_result: dict[str, Any]) -> HybridIndex:
    return tune_result["index"]
