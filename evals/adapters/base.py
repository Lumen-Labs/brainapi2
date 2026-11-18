from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class QueryResult:
    answer: str
    passages: list[dict[str, Any]] = field(default_factory=list)
    conflict_pairs: list[tuple[str, str]] = field(default_factory=list)
    retrieve_ms: float = 0.0
    answer_ms: float = 0.0
    e2e_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    llm_calls: int = 0
    http_calls: int = 0
    raw: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


class SystemAdapter(Protocol):
    name: str
    capabilities: dict[str, Any]

    def ingest(self, docs: list[dict[str, Any]]) -> dict[str, Any]:
        ...

    def query(self, question: dict[str, Any]) -> QueryResult:
        ...
