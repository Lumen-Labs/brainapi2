"""Local retrieve/context stage profiler with representative I/O delays.

Run:
    poetry run python tests/profile_context_retrieve.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ENV_DEFAULTS = {
    "BRAINPAT_TOKEN": "test-token",
    "MODELS_MODE": "local",
    "EMBEDDINGS_LOCAL_MODEL": "local-model",
    "EMBEDDINGS_SMALL_MODEL": "small-model",
    "EMBEDDING_NODES_DIMENSION": "3",
    "EMBEDDING_TRIPLETS_DIMENSION": "3",
    "EMBEDDING_OBSERVATIONS_DIMENSION": "3",
    "EMBEDDING_DATA_DIMENSION": "3",
    "EMBEDDING_RELATIONSHIPS_DIMENSION": "3",
    "REDIS_HOST": "localhost",
    "REDIS_PORT": "6379",
    "NEO4J_HOST": "localhost",
    "NEO4J_PORT": "7687",
    "NEO4J_USERNAME": "neo4j",
    "NEO4J_PASSWORD": "password",
    "MILVUS_HOST": "localhost",
    "MILVUS_PORT": "19530",
    "MONGO_CONNECTION_STRING": "mongodb://localhost:27017",
    "CELERY_WORKER_CONCURRENCY": "1",
    "OLLAMA_HOST": "localhost",
    "OLLAMA_PORT": "11434",
    "OLLAMA_LLM_SMALL_MODEL": "small",
    "OLLAMA_LLM_LARGE_MODEL": "large",
    "POSTGRES_HOST": "localhost",
    "POSTGRES_PORT": "5432",
    "POSTGRES_USERNAME": "postgres",
    "POSTGRES_PASSWORD": "postgres",
}
for key, value in ENV_DEFAULTS.items():
    os.environ.setdefault(key, value)

from src.constants.embeddings import Vector
from src.constants.kg import Node, Predicate
from src.core.saving.history_intent import reset_prototype_cache
from src.services.api.constants.requests import GetContextRequestBody
from src.utils.nlp.ner import ExtractElementsResponse

EMBED_TEXT_S = 0.25
EMBED_BATCH_S = 0.40
VECTOR_SEARCH_S = 0.03
GRAPH_S = 0.04
DATA_S = 0.02
QUERY = "Who is Alice and what happened at the event?"


def _sleep(seconds: float) -> None:
    if seconds > 0:
        time.sleep(seconds)


def _node(uuid: str, name: str, *, happened_at: str | None = None, labels=None):
    return Node(
        uuid=uuid,
        name=name,
        labels=labels or ["PERSON"],
        description=f"desc-{name}",
        happened_at=happened_at,
    )


def _pred(uuid: str, name: str):
    return Predicate(uuid=uuid, name=name, description=f"desc-{name}")


def _facts(count: int = 8):
    return [
        (
            _node("a", "Alice"),
            _pred(f"r{i}a", f"ATTENDED_{i}"),
            _node(
                f"e{i}",
                f"Event{i}",
                happened_at=f"2024-0{(i % 9) + 1}-01",
                labels=["EVENT"],
            ),
            _pred(f"r{i}b", f"WITH_{i}"),
            _node(f"b{i}", f"Person{i}"),
        )
        for i in range(count)
    ]


def _vector(uid: str, distance: float = 0.1) -> Vector:
    return Vector(
        id=uid,
        embeddings=[0.1, 0.2, 0.3],
        metadata={"uuid": uid, "name": uid, "resource_id": uid},
        distance=distance,
    )


class CallMeter:
    def __init__(self):
        self.counts: Counter[str] = Counter()

    def mark(self, name: str) -> None:
        self.counts[name] += 1


def _build_mocks(meter: CallMeter):
    facts = _facts(8)

    def embed_text(text: str):
        meter.mark("embed_text")
        _sleep(EMBED_TEXT_S)
        return Vector(id="q", embeddings=[0.1, 0.2, 0.3], metadata={})

    def embed_texts(texts):
        meter.mark("embed_texts")
        meter.mark("embed_texts_items")
        meter.counts["embed_texts_items"] += max(len(texts) - 1, 0)
        _sleep(EMBED_BATCH_S)
        return [
            Vector(id=f"b{i}", embeddings=[0.1, 0.2, 0.3], metadata={})
            for i, _ in enumerate(texts)
        ]

    embeddings = MagicMock()
    embeddings.embed_text.side_effect = embed_text
    embeddings.embed_texts.side_effect = embed_texts

    def search_nodes(*_args, **_kwargs):
        meter.mark("search_nodes")
        _sleep(VECTOR_SEARCH_S)
        return [_vector("a", 0.1)]

    def search_relationships(*_args, **_kwargs):
        meter.mark("search_relationships")
        _sleep(VECTOR_SEARCH_S)
        return []

    def search_data(*_args, **_kwargs):
        meter.mark("search_data")
        _sleep(VECTOR_SEARCH_S)
        return [_vector("c1", 0.2)]

    vs = MagicMock()
    vs.search_nodes.side_effect = search_nodes
    vs.search_relationships.side_effect = search_relationships
    vs.search_data.side_effect = search_data

    def event_neighbors(*_args, **_kwargs):
        meter.mark("graph_neighbors")
        _sleep(GRAPH_S)
        return facts

    graph = MagicMock()
    graph.get_event_centric_neighbors.side_effect = event_neighbors
    graph.get_hub_bridges.return_value = []
    graph.list_topic_memberships.return_value = []

    def fetch_chunks(*_args, **_kwargs):
        meter.mark("fetch_chunks")
        _sleep(DATA_S)
        return ([], [])

    def keyword_search(*_args, **_kwargs):
        meter.mark("keyword_search")
        _sleep(DATA_S)
        return SimpleNamespace(text_chunks=[])

    data = MagicMock()
    data.get_last_text_chunks.return_value = []
    data.get_last_structured_data.return_value = []
    data.search.side_effect = keyword_search
    data.get_text_chunks_by_ids.side_effect = fetch_chunks
    return embeddings, vs, graph, data


async def _run_once():
    from src.services.api.controllers import retrieve as retrieve_mod

    reset_prototype_cache()
    meter = CallMeter()
    embeddings, vs, graph, data = _build_mocks(meter)
    elements = ExtractElementsResponse(
        tokens=[
            {"text": "Alice", "lemma": "Alice", "pos": "PROPN"},
            {"text": "event", "lemma": "event", "pos": "NOUN"},
        ],
        noun_chunks=["the event"],
    )
    started = time.perf_counter()
    with (
        patch.object(retrieve_mod, "embeddings_adapter", embeddings),
        patch.object(retrieve_mod, "vector_search", vs),
        patch.object(retrieve_mod, "graph_adapter", graph),
        patch.object(retrieve_mod, "data_adapter", data),
        patch.object(
            retrieve_mod._entity_extractor,
            "extract_elements",
            return_value=elements,
        ),
    ):
        response = await retrieve_mod.get_context(
            GetContextRequestBody(
                text=QUERY,
                brain_id="qa-small",
                profile_stages=True,
            )
        )
    wall_ms = (time.perf_counter() - started) * 1000
    return response, wall_ms, meter


def _stages(report: dict) -> list[dict]:
    return list(report.get("stages") or [])


def main() -> int:
    response, wall_ms, meter = asyncio.run(_run_once())
    report = response.stage_timings or {}
    payload = {
        "label": "local fixture with representative I/O delays",
        "query_kind": "small QA brain, ~8 event facts, no live network",
        "delays_s": {
            "embed_text": EMBED_TEXT_S,
            "embed_batch": EMBED_BATCH_S,
            "vector_search": VECTOR_SEARCH_S,
            "graph": GRAPH_S,
            "data": DATA_S,
        },
        "client_wall_ms": round(wall_ms, 3),
        "reported_total_ms": report.get("total_ms"),
        "calls": dict(meter.counts),
        "stages": [
            {
                "stage": item.get("stage"),
                "wall_ms": item.get("wall_ms"),
                "calls": item.get("calls"),
                "overlap_ms": item.get("overlap_ms"),
            }
            for item in _stages(report)
        ],
    }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
