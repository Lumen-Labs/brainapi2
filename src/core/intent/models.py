from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

FACT_STATUSES = ("verified", "unverified", "rejected", "superseded")
FactStatus = Literal["verified", "unverified", "rejected", "superseded"]

CHECK_VERDICTS = ("supported", "contradicted", "superseded", "unknown")
CheckVerdict = Literal["supported", "contradicted", "superseded", "unknown"]

ResponseFormat = Literal["concise", "detailed"]


class FactSource(BaseModel):
    doc: Optional[str] = None
    url: Optional[str] = None
    source_key: Optional[str] = None
    writer: Optional[str] = None


class IntentFact(BaseModel):
    fact_id: str
    text: str
    status: FactStatus
    subject: str = ""
    predicate: str = ""
    object: str = ""
    contradicted_by: list[str] = Field(default_factory=list)
    valid_at: Optional[str] = None
    invalid_at: Optional[str] = None
    superseded_by: Optional[str] = None
    source: FactSource = Field(default_factory=FactSource)
    writer: Optional[str] = None
    retired_reason: Optional[str] = None
    verification_state: Optional[str] = None
    kind: Literal["relationship", "node", "passage"] = "relationship"

    def natural_id(self) -> str:
        if self.subject and self.predicate and self.object:
            return f"{self.subject} {self.predicate} {self.object}"
        return self.text


class RecallResult(BaseModel):
    text: str
    as_of: Optional[str] = None
    brain_id: str = ""
    facts: list[IntentFact] = Field(default_factory=list)
    matched: bool = False
    total: int = 0
    skip: int = 0
    limit: int = 8
    next_skip: Optional[int] = None
    truncated: bool = False
    error: Optional[str] = None
    code: Optional[str] = None
    hint: Optional[str] = None


class CheckResult(BaseModel):
    text: str
    verdict: CheckVerdict = "unknown"
    claim: str = ""
    as_of: Optional[str] = None
    brain_id: str = ""
    evidence: list[IntentFact] = Field(default_factory=list)
    matched: bool = False
    error: Optional[str] = None
    code: Optional[str] = None
    hint: Optional[str] = None


class RememberResult(BaseModel):
    text: str
    brain_id: str = ""
    fact: Optional[IntentFact] = None
    superseded: list[str] = Field(default_factory=list)
    error: Optional[str] = None
    code: Optional[str] = None
    hint: Optional[str] = None


class CorrectResult(BaseModel):
    text: str
    brain_id: str = ""
    fact: Optional[IntentFact] = None
    previous: Optional[IntentFact] = None
    superseded: list[str] = Field(default_factory=list)
    error: Optional[str] = None
    code: Optional[str] = None
    hint: Optional[str] = None


class IntentError(BaseModel):
    error: str
    code: str
    hint: Optional[str] = None

    def as_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"error": self.error, "code": self.code}
        if self.hint:
            payload["hint"] = self.hint
        return payload
