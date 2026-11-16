"""Cross-source fact conflict records.

A conflict links two facts that share a subject+predicate claim slot, disagree
on value/amount/polarity, and are valid at the same time. Same-source
revisions (superseded v1 vs current v2) are not conflicts.

`verification_state` on each side is reserved for a parallel fact-state
change: verifying one side should resolve the conflict via
`resolve_by_verified_fact` without this module implementing verification.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

from src.core.saving.validity import format_instant, utc_now

ConflictStatus = Literal["open", "resolved", "dismissed"]
DisagreementType = Literal[
    "value_mismatch",
    "numeric_mismatch",
    "polarity_conflict",
    "semantic_contradiction",
]
DetectionMethod = Literal["deterministic", "llm"]
VerificationState = Literal["unverified", "verified", "rejected"]
ResolutionMethod = Literal["dismissed", "verified_side", "manual"]


class ConflictSource(BaseModel):
    chunk_ids: list[str] = Field(default_factory=list)
    source_key: Optional[str] = None
    document: Optional[str] = None
    timestamp: Optional[str] = None
    text: Optional[str] = None


class ConflictEntity(BaseModel):
    uuid: str = ""
    name: str = ""
    labels: list[str] = Field(default_factory=list)


class ConflictSide(BaseModel):
    fact_id: str
    subject: ConflictEntity
    predicate: str
    object: ConflictEntity
    value: str = ""
    polarity: str = "neutral"
    amount: Optional[float] = None
    valid_from: Optional[str] = None
    valid_to: Optional[str] = None
    superseded_by: Optional[str] = None
    verification_state: VerificationState = "unverified"
    source: ConflictSource = Field(default_factory=ConflictSource)


class ConflictResolution(BaseModel):
    method: Optional[ResolutionMethod] = None
    winning_fact_id: Optional[str] = None
    notes: Optional[str] = None
    resolved_at: Optional[str] = None
    resolved_by: Optional[str] = None


class FactConflict(BaseModel):
    id: str
    pair_key: str
    claim_key: str
    status: ConflictStatus = "open"
    disagreement_type: DisagreementType
    confidence: float = 1.0
    detection_method: DetectionMethod = "deterministic"
    detected_at: str = Field(default_factory=lambda: format_instant(utc_now()))
    subject: ConflictEntity
    predicate: str
    sides: list[ConflictSide]
    resolution: ConflictResolution = Field(default_factory=ConflictResolution)

    def source_keys(self) -> list[str]:
        keys: list[str] = []
        seen: set[str] = set()
        for side in self.sides:
            key = side.source.source_key
            if key and key not in seen:
                seen.add(key)
                keys.append(key)
        return keys

    def fact_ids(self) -> list[str]:
        return [side.fact_id for side in self.sides if side.fact_id]


def conflict_to_document(conflict: FactConflict) -> dict[str, Any]:
    return conflict.model_dump(mode="json")


def conflict_from_document(document: dict[str, Any] | None) -> FactConflict | None:
    if not document:
        return None
    payload = dict(document)
    payload.pop("_id", None)
    return FactConflict.model_validate(payload)
