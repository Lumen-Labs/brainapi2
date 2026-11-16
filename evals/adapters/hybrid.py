from __future__ import annotations

import time
from typing import Any

from evals.adapters.base import QueryResult
from evals.answer import extractive_answer, llm_answer, pack_context
from evals.baseline.contradict import detect_conflicts
from evals.baseline.dates import filter_as_of
from evals.baseline.dense import make_embedder
from evals.baseline.index import DEFAULT_HYBRID_CONFIG, HybridIndex, build_index
from evals.baseline.rerank import CrossEncoderReranker
from evals.baseline.textlib import sentences
from evals.config import Settings, PROMPT_BUDGET


def _diff_versions(d1_docs: list[dict[str, Any]], d2_docs: list[dict[str, Any]]) -> str:
    def by_clause(docs: list[dict[str, Any]]) -> dict[str, str]:
        out: dict[str, str] = {}
        for doc in docs:
            key = str(doc.get("clause_id") or doc.get("id"))
            out[key] = str(doc.get("text") or "")
        return out

    left = by_clause(d1_docs)
    right = by_clause(d2_docs)
    lines: list[str] = []
    for key in sorted(set(left) | set(right)):
        a = set(sentences(left.get(key, "")))
        b = set(sentences(right.get(key, "")))
        removed = sorted(a - b)
        added = sorted(b - a)
        if not removed and not added:
            continue
        lines.append(f"Clause {key}:")
        for sent in removed:
            lines.append(f"- {sent}")
        for sent in added:
            lines.append(f"+ {sent}")
    return "\n".join(lines) if lines else "No textual differences retrieved."


class HybridAdapter:
    name = "hybrid_baseline"

    def __init__(
        self,
        settings: Settings,
        *,
        config: dict[str, Any] | None = None,
        answerer: str = "extractive",
    ) -> None:
        self.settings = settings
        self.config = dict(DEFAULT_HYBRID_CONFIG)
        if config:
            self.config.update(config)
        self.answerer = answerer
        self.index: HybridIndex | None = None
        self.docs: list[dict[str, Any]] = []
        self.ingest_stats: dict[str, Any] = {}
        self.capabilities = {
            "as_of": True,
            "as_of_mode": "validity_interval_filter",
            "conflict_pairs": True,
            "conflict_mode": "retrieval_plus_numeric_negation_heuristic",
            "pluggable": False,
            "status": "ready",
        }

    def ingest(self, docs: list[dict[str, Any]]) -> dict[str, Any]:
        self.docs = list(docs)
        embedder = make_embedder(
            self.settings.embedding_provider
            if self.settings.embedding_provider != "hash"
            else str(self.config.get("dense") or "hash"),
            api_key=self.settings.embedding_api_key,
            model=self.settings.embedding_model,
            base_url=self.settings.embedding_base_url,
            dim=int(self.config.get("dense_dim") or 256),
        )
        cross = None
        if self.settings.reranker in {"cross-encoder", "ce", "cross_encoder"}:
            try:
                cross = CrossEncoderReranker(self.settings.cross_encoder_model)
                self.config["reranker"] = "cross-encoder"
            except Exception as exc:
                self.config["reranker"] = "lexical"
                self.capabilities["reranker_fallback"] = f"lexical ({exc.__class__.__name__})"
        started = time.perf_counter()
        self.index = build_index(
            docs, self.config, embedder=embedder, cross_encoder=cross
        )
        elapsed = (time.perf_counter() - started) * 1000
        self.ingest_stats = {
            "docs": len(docs),
            "chunks": len(self.index.chunks),
            "ingest_ms": elapsed,
            "cost": dict(self.index.cost),
            "config": dict(self.config),
        }
        return self.ingest_stats

    def query(self, question: dict[str, Any]) -> QueryResult:
        if self.index is None:
            raise RuntimeError("HybridAdapter.ingest must run before query")
        started = time.perf_counter()
        qtext = str(question.get("question") or "")
        as_of = question.get("as_of")
        notes: list[str] = []
        passages: list[dict[str, Any]] = []
        pairs: list[tuple[str, str]] = []
        retrieve_started = time.perf_counter()
        if question.get("task") == "changelog":
            d1 = question.get("date_from")
            d2 = question.get("date_to")
            hits1 = self.index.search(qtext, k=int(PROMPT_BUDGET["max_passages"]), as_of=d1)
            hits2 = self.index.search(qtext, k=int(PROMPT_BUDGET["max_passages"]), as_of=d2)
            passages = hits1 + hits2
            diff = _diff_versions(filter_as_of(self.docs, d1), filter_as_of(self.docs, d2))
            notes.append("changelog_diff_from_validity_filter")
            extract_seed = [{"text": diff, "title": "diff", "doc_id": "diff"}] + passages
            retrieve_ms = (time.perf_counter() - retrieve_started) * 1000
            answer = self._answer(question, extract_seed, fallback=diff)
        elif question.get("task") == "conflict_audit":
            threshold = float(self.config.get("conflict_threshold") or 0.5)
            detected = detect_conflicts(
                self.index, self.docs, threshold=threshold, neighbors=8
            )
            pairs = [(row["left"], row["right"]) for row in detected]
            passages = [
                {
                    "id": f"{row['left']}::{row['right']}",
                    "doc_id": row["left"],
                    "text": f"Conflict {row['left']} vs {row['right']} score={row['score']:.3f}",
                    "title": "conflict",
                }
                for row in detected
            ]
            retrieve_ms = (time.perf_counter() - retrieve_started) * 1000
            listed = ", ".join(f"{a} vs {b}" for a, b in pairs) or "none"
            answer = self._answer(
                question,
                passages,
                fallback=f"Conflicting statement pairs: {listed}",
            )
        else:
            passages = self.index.search(
                qtext, k=int(PROMPT_BUDGET["max_passages"]), as_of=as_of
            )
            retrieve_ms = (time.perf_counter() - retrieve_started) * 1000
            answer = self._answer(question, passages)
        e2e_ms = (time.perf_counter() - started) * 1000
        return QueryResult(
            answer=answer["text"],
            passages=passages,
            conflict_pairs=pairs,
            retrieve_ms=retrieve_ms,
            answer_ms=answer["latency_ms"],
            e2e_ms=e2e_ms,
            prompt_tokens=answer["prompt_tokens"],
            completion_tokens=answer["completion_tokens"],
            total_tokens=answer["total_tokens"],
            llm_calls=answer["calls"],
            http_calls=0,
            raw={"packed_context": pack_context(passages, self.settings)},
            notes=notes,
        )

    def _answer(
        self,
        question: dict[str, Any],
        passages: list[dict[str, Any]],
        fallback: str | None = None,
    ) -> dict[str, Any]:
        if self.answerer == "llm":
            result = llm_answer(
                self.settings, str(question.get("question") or ""), passages
            )
            return {
                "text": result.text,
                "latency_ms": result.latency_ms,
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "total_tokens": result.total_tokens,
                "calls": result.calls,
            }
        result = extractive_answer(question, passages)
        text = result.text or (fallback or "")
        return {
            "text": text,
            "latency_ms": 0.0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "calls": 0,
        }
