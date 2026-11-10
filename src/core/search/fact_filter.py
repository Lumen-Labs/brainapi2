from __future__ import annotations

from typing import Any, Optional

import numpy as np
from pydantic import BaseModel, Field


class FactFilterDecision(BaseModel):
    keep_indices: list[int] = Field(default_factory=list)


def filter_relevant_facts(
    question: str,
    candidates: list[str],
    *,
    llm_adapter: Any = None,
    max_keep: int = 40,
) -> list[int]:
    if not candidates:
        return []

    # Without an LLM, preserve recall: return every candidate index and let the
    # caller apply score-based capping. Truncating here would drop later
    # high-value evidence that ranking already ordered.
    if llm_adapter is None:
        return list(range(len(candidates)))

    indexed = "\n".join(f"{i}: {text}" for i, text in enumerate(candidates))
    prompt = (
        "You are a recognition-memory filter for a conversational memory system.\n"
        "Given a question and numbered candidate facts/passages, return the indices "
        "of items that are relevant evidence for answering the question.\n"
        "Prefer recall of useful evidence over aggressive pruning, but drop clear noise.\n"
        f"Keep at most {max_keep} indices.\n\n"
        f"Question: {question}\n\n"
        f"Candidates:\n{indexed}\n\n"
        'Respond with JSON: {"keep_indices": [0, 2, ...]}'
    )

    try:
        response = llm_adapter.generate_structured(
            prompt,
            FactFilterDecision,
        )
        if isinstance(response, FactFilterDecision):
            indices = response.keep_indices
        elif isinstance(response, dict):
            indices = response.get("keep_indices") or []
        else:
            indices = getattr(response, "keep_indices", []) or []
        cleaned = sorted(
            {
                int(i)
                for i in indices
                if isinstance(i, (int, float, str))
                and str(i).isdigit()
                and 0 <= int(i) < len(candidates)
            }
        )
        if cleaned:
            return cleaned[:max_keep]
    except Exception:
        pass
    return list(range(len(candidates)))


def reciprocal_rank_fusion(
    ranked_lists: list[list[str]],
    *,
    k: int = 60,
) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, item in enumerate(ranked):
            if not item:
                continue
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda pair: pair[1], reverse=True)


def _normalize_weighted_neighbors(
    neighbors: list | dict | None,
) -> list[tuple[str, float]]:
    if not neighbors:
        return []
    if isinstance(neighbors, dict):
        items = [(str(dst), float(weight)) for dst, weight in neighbors.items()]
    else:
        items = []
        for entry in neighbors:
            if isinstance(entry, tuple) and len(entry) == 2:
                items.append((str(entry[0]), float(entry[1])))
            else:
                items.append((str(entry), 1.0))
    cleaned: list[tuple[str, float]] = []
    for dst, weight in items:
        if not dst:
            continue
        cleaned.append((dst, max(0.0, weight)))
    return cleaned


def personalized_pagerank(
    adjacency: dict[str, list | dict],
    seeds: dict[str, float],
    *,
    damping: float = 0.85,
    iterations: int = 20,
) -> dict[str, float]:
    if not adjacency and not seeds:
        return {}
    weighted: dict[str, list[tuple[str, float]]] = {}
    nodes = set(seeds.keys())
    for src, neighbors in adjacency.items():
        src_key = str(src)
        nodes.add(src_key)
        edges = _normalize_weighted_neighbors(neighbors)
        weighted[src_key] = edges
        for dst, _weight in edges:
            nodes.add(dst)
    if not nodes:
        return {}
    node_list = sorted(nodes)
    seed_mass = sum(max(0.0, w) for w in seeds.values()) or 1.0
    personalization = {
        node: (max(0.0, seeds.get(node, 0.0)) / seed_mass) for node in node_list
    }
    if sum(personalization.values()) <= 0:
        uniform = 1.0 / len(node_list)
        personalization = {node: uniform for node in node_list}
    node_index = {node: index for index, node in enumerate(node_list)}
    sources, targets, weights, dangling = [], [], [], []
    for source, node in enumerate(node_list):
        neighbors = weighted.get(node) or []
        total = sum(weight for _, weight in neighbors)
        if total <= 0:
            dangling.append(source)
            continue
        for target, weight in neighbors:
            sources.append(source)
            targets.append(node_index[target])
            weights.append(weight / total)
    source_indices = np.asarray(sources, dtype=np.intp)
    target_indices = np.asarray(targets, dtype=np.intp)
    dangling_indices = np.asarray(dangling, dtype=np.intp)
    transition_weights = np.asarray(weights, dtype=np.float64)
    scores = np.fromiter(personalization.values(), dtype=np.float64)
    teleport = (1.0 - damping) * scores
    for _ in range(max(1, iterations)):
        next_scores = teleport.copy()
        # Keep edge accumulation order so equal scores retain the old tie breaks.
        np.add.at(
            next_scores,
            target_indices,
            (damping * scores[source_indices]) * transition_weights,
        )
        dangling_mass = sum(scores[dangling_indices])
        if dangling_mass > 0:
            next_scores += damping * dangling_mass / len(node_list)
        scores = next_scores
    return dict(zip(node_list, scores.tolist()))
