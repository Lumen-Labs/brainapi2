#!/usr/bin/env python3
"""Replay the public docs question through the real controller with fixture adapters.

Storage delays and evidence are synthetic. This measures controller work and
evidence preservation, not production endpoint latency or general relevance.
"""
import argparse
import asyncio
import hashlib
import json
import math
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from threading import BoundedSemaphore, Lock
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from test_stage_latency_profiler import (  # noqa: E402
    ExtractElementsResponse,
    GetContextRequestBody,
    Vector,
    _node,
    _pred,
    _stubbed_adapters,
)

QUESTION = "How does BrainAPI store, organize, and retrieve memories and contextual information?"
PASSAGES = [
    "BrainAPI stores memories as source passages and embeddings, keeping the original contextual information available for retrieval.",
    "BrainAPI organizes memories into entities and relationships in a knowledge graph, linking facts to source passages.",
    "To retrieve memories, BrainAPI combines vector similarity with graph context and returns supporting passages.",
    "The context endpoint assembles contextual information from ranked facts and source passages so answers can cite their evidence.",
    "A brain separates memory data from other brains and preserves source references during ingestion.",
    "Structured facts describe how entities relate; text passages preserve the original wording.",
    "Vector search retrieves semantically related information even when the wording differs.",
    "Graph ranking organizes candidates by connections to the question’s entities.",
]


async def replay(iterations: int) -> dict:
    from src.services.api.controllers import retrieve

    facts = [
        (
            _node(f"doc-{i}", "BrainAPI" if not i else f"BrainAPI memory component {i}"),
            _pred(f"r{i}a", ("STORES", "ORGANIZES", "RETRIEVES")[i % 3]),
            _node(f"memory-{i}", f"Memory and contextual information {i}"),
            _pred(f"r{i}b", "REFERENCES"),
            _node(f"passage-{i}", f"Source passage {i}"),
        )
        for i in range(44)
    ]
    samples = []
    for _ in range(iterations):
        counts: Counter = Counter()
        lock = Lock()
        vector_pool = BoundedSemaphore(2)

        def count(name):
            with lock:
                counts[name] += 1

        def embed(text):
            count("embeddings")
            time.sleep(.004)
            return Vector(id=text, embeddings=[.1, .2, .3], metadata={})

        def search(name, result):
            def run(*args, **kwargs):
                count(name)
                with vector_pool:
                    time.sleep(.022)
                return result
            return run

        seeds = [Vector(id="docs", embeddings=[.1, .2, .3], metadata={"uuid": "doc-0"}, distance=.1)]
        passages = [(str(i), i / 100, body) for i, body in enumerate(PASSAGES)]
        config = [
            patch.object(retrieve, "embeddings_adapter", MagicMock(embed_text=embed)),
            patch.object(retrieve, "vector_search", MagicMock(
                search_nodes=search("node_searches", seeds),
                search_relationships=search("relationship_searches", []),
            )),
            patch.object(retrieve, "_retrieve_passages", return_value=passages),
        ]
        with _stubbed_adapters(retrieve, facts, extra=config):
            with patch.object(retrieve._entity_extractor, "extract_elements", return_value=ExtractElementsResponse(
                tokens=[{"text": text, "lemma": text, "pos": "NOUN"} for text in
                        ("BrainAPI", "store", "organize", "retrieve", "memories", "contextual information", "information", "context", "memory")],
                noun_chunks=[],
            )):
                response = await retrieve.get_context(GetContextRequestBody(
                    text=QUESTION, brain_id="balanced-replay", use_ppr=True,
                    max_facts=40, max_passages=8, profile_stages=True,
                ))
        profile = response.stage_timings
        found = response.source_passages
        samples.append({
            "ms": profile["total_ms"],
            "loop_blocked_ms": profile["loop_blocked_ms"],
            "calls": dict(counts),
            "passage_count": len(found),
            "gold_passages": sum(p in found for p in PASSAGES),
            "fact_count": len(response.triples),
            "source_digest": hashlib.sha256(json.dumps(found).encode()).hexdigest(),
            "fact_digest": hashlib.sha256(
                json.dumps([[item.uuid for item in fact.triple] for fact in response.triples]).encode()
            ).hexdigest(),
            "stages": {s["stage"]: s["wall_ms"] for s in profile["stages"] if s["stage"] in {"retrieval.fanout", "ppr"}},
        })
    ordered = sorted(s["ms"] for s in samples)
    return {"question": QUESTION, "iterations": iterations, "p50_ms": statistics.median(ordered),
            "p95_ms": ordered[math.ceil(.95 * len(ordered)) - 1], "samples": samples}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=10)
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("iterations must be positive")
    print(json.dumps(asyncio.run(replay(args.iterations)), indent=2))
