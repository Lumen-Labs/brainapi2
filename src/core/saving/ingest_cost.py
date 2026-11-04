from __future__ import annotations

import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Iterator, Optional

SOURCE_TOKENIZER_ID = "tiktoken/cl100k_base"
_SOURCE_ENCODING = None
_SOURCE_ENCODING_FAILED = False


def _get_cl100k_encoding():
    global _SOURCE_ENCODING, _SOURCE_ENCODING_FAILED
    if _SOURCE_ENCODING is not None:
        return _SOURCE_ENCODING
    if _SOURCE_ENCODING_FAILED:
        return None
    try:
        import tiktoken

        _SOURCE_ENCODING = tiktoken.get_encoding("cl100k_base")
        return _SOURCE_ENCODING
    except Exception:
        _SOURCE_ENCODING_FAILED = True
        return None


def count_source_tokens(text: str) -> tuple[int, str, bool]:
    """
    Count source tokens with the pinned cross-provider tokenizer.

    Returns (token_count, tokenizer_id, estimated).
    """
    if not text:
        return 0, SOURCE_TOKENIZER_ID, False
    enc = _get_cl100k_encoding()
    if enc is not None:
        return len(enc.encode(text)), SOURCE_TOKENIZER_ID, False
    return max(1, len(text) // 4), "estimate/chars_div_4", True


@dataclass
class StageCost:
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    latency_ms: float = 0.0
    calls: int = 0

    def add_usage(
        self,
        *,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cached_tokens: int = 0,
        latency_ms: float = 0.0,
        calls: int = 1,
    ) -> None:
        self.input_tokens += int(input_tokens or 0)
        self.output_tokens += int(output_tokens or 0)
        self.cached_tokens += int(cached_tokens or 0)
        self.latency_ms += float(latency_ms or 0.0)
        self.calls += int(calls or 0)

    def merge(self, other: Optional["StageCost"]) -> None:
        if other is None:
            return
        self.add_usage(
            input_tokens=other.input_tokens,
            output_tokens=other.output_tokens,
            cached_tokens=other.cached_tokens,
            latency_ms=other.latency_ms,
            calls=other.calls,
        )

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cached_tokens": self.cached_tokens,
            "total_tokens": self.total_tokens,
            "latency_ms": round(self.latency_ms, 3),
            "calls": self.calls,
        }


@dataclass
class IngestCostLedger:
    scout: StageCost = field(default_factory=StageCost)
    architect: StageCost = field(default_factory=StageCost)
    janitor: StageCost = field(default_factory=StageCost)
    observations: StageCost = field(default_factory=StageCost)
    consolidation: StageCost = field(default_factory=StageCost)
    embed: StageCost = field(default_factory=StageCost)
    index_rebuild: StageCost = field(default_factory=StageCost)
    janitor_skipped: int = 0
    janitor_ran: int = 0
    janitor_rejected: int = 0
    janitor_ambiguous: int = 0
    janitor_drop_reasons: list[str] = field(default_factory=list)
    er_adjudications: int = 0
    source_chars: int = 0
    source_tokens: int = 0
    source_tokenizer: str = SOURCE_TOKENIZER_ID
    source_tokens_estimated: bool = False
    architect_units: int = 0
    architect_escalations: int = 0
    architect_schema_calls: int = 0
    architect_repair_calls: int = 0
    escalate_reasons: list[str] = field(default_factory=list)

    def stage(self, name: str) -> StageCost:
        return getattr(self, name)

    def set_source_text(self, text: str) -> None:
        self.source_chars = len(text or "")
        tokens, tokenizer_id, estimated = count_source_tokens(text or "")
        self.source_tokens = tokens
        self.source_tokenizer = tokenizer_id
        self.source_tokens_estimated = estimated

    def record_architect_unit(
        self,
        *,
        escalated: bool = False,
        reason: Optional[str] = None,
        schema_calls: int = 0,
        repair_calls: int = 0,
    ) -> None:
        self.architect_units += 1
        self.architect_schema_calls += int(schema_calls or 0)
        self.architect_repair_calls += int(repair_calls or 0)
        if escalated:
            self.architect_escalations += 1
            if reason:
                self.escalate_reasons.append(reason)

    @property
    def llm_source_multiplier(self) -> Optional[float]:
        if self.source_tokens <= 0:
            return None
        return round(self.total_llm_tokens / float(self.source_tokens), 3)

    @property
    def escalate_rate(self) -> Optional[float]:
        if self.architect_units <= 0:
            return None
        return round(self.architect_escalations / float(self.architect_units), 4)

    @property
    def janitor_skip_rate(self) -> Optional[float]:
        """Share of edges auto-accepted without LLM Janitor."""
        if self.janitor_ambiguous or self.janitor_rejected:
            total = (
                self.janitor_skipped
                + self.janitor_rejected
                + self.janitor_ambiguous
            )
        else:
            total = self.janitor_skipped + self.janitor_ran
        if total <= 0:
            return None
        return round(self.janitor_skipped / float(total), 4)

    @property
    def total_llm_tokens(self) -> int:
        return (
            self.scout.total_tokens
            + self.architect.total_tokens
            + self.janitor.total_tokens
            + self.observations.total_tokens
            + self.consolidation.total_tokens
        )

    def merge(self, other: Optional["IngestCostLedger"]) -> None:
        if other is None:
            return
        self.scout.merge(other.scout)
        self.architect.merge(other.architect)
        self.janitor.merge(other.janitor)
        self.observations.merge(other.observations)
        self.consolidation.merge(other.consolidation)
        self.embed.merge(other.embed)
        self.index_rebuild.merge(other.index_rebuild)
        self.janitor_skipped += other.janitor_skipped
        self.janitor_ran += other.janitor_ran
        self.janitor_rejected += other.janitor_rejected
        self.janitor_ambiguous += other.janitor_ambiguous
        self.janitor_drop_reasons.extend(other.janitor_drop_reasons)
        self.er_adjudications += other.er_adjudications
        self.source_chars = max(self.source_chars, other.source_chars)
        if other.source_tokens and (
            not self.source_tokens or other.source_tokens > self.source_tokens
        ):
            self.source_tokens = other.source_tokens
            self.source_tokenizer = other.source_tokenizer
            self.source_tokens_estimated = other.source_tokens_estimated
        self.architect_units += other.architect_units
        self.architect_escalations += other.architect_escalations
        self.architect_schema_calls += other.architect_schema_calls
        self.architect_repair_calls += other.architect_repair_calls
        self.escalate_reasons.extend(other.escalate_reasons)

    def to_dict(self) -> dict[str, Any]:
        stages = {
            "scout": self.scout.to_dict(),
            "architect": self.architect.to_dict(),
            "janitor": self.janitor.to_dict(),
            "observations": self.observations.to_dict(),
            "consolidation": self.consolidation.to_dict(),
            "embed": self.embed.to_dict(),
            "index_rebuild": self.index_rebuild.to_dict(),
        }
        total_in = sum(s["input_tokens"] for s in stages.values())
        total_out = sum(s["output_tokens"] for s in stages.values())
        total_cached = sum(s["cached_tokens"] for s in stages.values())
        multiplier = self.llm_source_multiplier
        return {
            **{f"{k}_{field}": v[field] for k, v in stages.items() for field in v},
            "stages": stages,
            "total_input_tokens": total_in,
            "total_output_tokens": total_out,
            "total_cached_tokens": total_cached,
            "total_llm_tokens": total_in + total_out,
            "janitor_skipped": self.janitor_skipped,
            "janitor_ran": self.janitor_ran,
            "janitor_rejected": self.janitor_rejected,
            "janitor_ambiguous": self.janitor_ambiguous,
            "janitor_drop_reasons": list(self.janitor_drop_reasons),
            "janitor_skip_rate": self.janitor_skip_rate,
            "er_adjudications": self.er_adjudications,
            "source_chars": self.source_chars,
            "source_tokens": self.source_tokens,
            "source_tokenizer": self.source_tokenizer,
            "source_tokens_estimated": self.source_tokens_estimated,
            "llm_source_multiplier": multiplier,
            "architect_units": self.architect_units,
            "architect_escalations": self.architect_escalations,
            "architect_schema_calls": self.architect_schema_calls,
            "architect_repair_calls": self.architect_repair_calls,
            "escalate_rate": self.escalate_rate,
            "escalate_reasons": list(self.escalate_reasons),
            "calls_per_unit": (
                round(self.architect.calls / float(self.architect_units), 3)
                if self.architect_units
                else None
            ),
        }

    @classmethod
    def from_dict(cls, data: Optional[dict[str, Any]]) -> "IngestCostLedger":
        ledger = cls()
        if not data:
            return ledger
        stages = data.get("stages") or {}
        for name in (
            "scout",
            "architect",
            "janitor",
            "observations",
            "consolidation",
            "embed",
            "index_rebuild",
        ):
            raw = stages.get(name) or {}
            stage = getattr(ledger, name)
            stage.input_tokens = int(raw.get("input_tokens") or 0)
            stage.output_tokens = int(raw.get("output_tokens") or 0)
            stage.cached_tokens = int(raw.get("cached_tokens") or 0)
            stage.latency_ms = float(raw.get("latency_ms") or 0.0)
            stage.calls = int(raw.get("calls") or 0)
        ledger.janitor_skipped = int(data.get("janitor_skipped") or 0)
        ledger.janitor_ran = int(data.get("janitor_ran") or 0)
        ledger.janitor_rejected = int(data.get("janitor_rejected") or 0)
        ledger.janitor_ambiguous = int(data.get("janitor_ambiguous") or 0)
        ledger.janitor_drop_reasons = list(data.get("janitor_drop_reasons") or [])
        ledger.er_adjudications = int(data.get("er_adjudications") or 0)
        ledger.source_chars = int(data.get("source_chars") or 0)
        ledger.source_tokens = int(data.get("source_tokens") or 0)
        ledger.source_tokenizer = str(
            data.get("source_tokenizer") or SOURCE_TOKENIZER_ID
        )
        ledger.source_tokens_estimated = bool(
            data.get("source_tokens_estimated") or False
        )
        ledger.architect_units = int(data.get("architect_units") or 0)
        ledger.architect_escalations = int(data.get("architect_escalations") or 0)
        ledger.architect_schema_calls = int(data.get("architect_schema_calls") or 0)
        ledger.architect_repair_calls = int(data.get("architect_repair_calls") or 0)
        reasons = data.get("escalate_reasons") or []
        if isinstance(reasons, list):
            ledger.escalate_reasons = [str(r) for r in reasons]
        return ledger


_active_stage: ContextVar[Optional[StageCost]] = ContextVar(
    "ingest_cost_active_stage", default=None
)
_active_ledger: ContextVar[Optional[IngestCostLedger]] = ContextVar(
    "ingest_cost_active_ledger", default=None
)


def get_active_ledger() -> Optional[IngestCostLedger]:
    return _active_ledger.get()


def get_active_stage() -> Optional[StageCost]:
    return _active_stage.get()


def record_usage_from_response(response: Any, latency_ms: float = 0.0) -> None:
    stage = _active_stage.get()
    if stage is None:
        return
    usage = getattr(response, "usage_metadata", None)
    if usage is None and isinstance(response, dict):
        usage = response.get("usage_metadata")
    if not usage and isinstance(response, dict):
        rm = response.get("response_metadata") or {}
        if isinstance(rm, dict):
            usage = rm.get("usage_metadata") or rm.get("token_usage")
    if not usage:
        rm = getattr(response, "response_metadata", None) or {}
        if isinstance(rm, dict):
            usage = rm.get("usage_metadata") or rm.get("token_usage")
    if not usage:
        return
    if not isinstance(usage, dict):
        usage = {
            "input_tokens": getattr(usage, "input_tokens", None)
            or getattr(usage, "prompt_tokens", 0),
            "output_tokens": getattr(usage, "output_tokens", None)
            or getattr(usage, "completion_tokens", 0),
        }
    details = usage.get("input_token_details") or {}
    cached = (
        details.get("cache_read", 0)
        or details.get("cached_tokens", 0)
        or usage.get("cached_tokens", 0)
        or 0
    )
    input_tokens = int(
        usage.get("input_tokens")
        or usage.get("prompt_tokens")
        or 0
    )
    output_tokens = int(
        usage.get("output_tokens")
        or usage.get("completion_tokens")
        or 0
    )
    stage.add_usage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached_tokens=int(cached or 0),
        latency_ms=latency_ms,
        calls=1,
    )


def submit_with_context(executor, fn, *args, **kwargs):
    """Propagate ContextVars (ingest cost stage) into ThreadPoolExecutor workers."""
    import contextvars

    ctx = contextvars.copy_context()
    return executor.submit(ctx.run, fn, *args, **kwargs)


def estimate_tokens(text: str) -> int:
    """Legacy char/4 estimate. Prefer count_source_tokens for ledger denominators."""
    if not text:
        return 0
    return max(1, len(text) // 4)


@contextmanager
def track_stage(ledger: IngestCostLedger, name: str) -> Iterator[StageCost]:
    stage = ledger.stage(name)
    from src.core.saving.pipeline_runtime import _stage
    token_name = _stage.set(name)
    from src.core.saving.pipeline_runtime import _context
    context = _context.get()
    if context and context.get("section_id"):
        from src.core.saving.ingest_manifest import update_section
        update_section(context["brain_id"], context["task_id"], "started", stage=name)
    token_stage = _active_stage.set(stage)
    token_ledger = _active_ledger.set(ledger)
    started = time.perf_counter()
    try:
        yield stage
    finally:
        stage.latency_ms += (time.perf_counter() - started) * 1000.0
        _stage.reset(token_name)
        _active_stage.reset(token_stage)
        _active_ledger.reset(token_ledger)



@contextmanager
def bind_ledger(ledger: IngestCostLedger) -> Iterator[IngestCostLedger]:
    token = _active_ledger.set(ledger)
    try:
        yield ledger
    finally:
        _active_ledger.reset(token)


def merge_cost_into_status_payload(
    existing: dict[str, Any], ledger: Optional[IngestCostLedger]
) -> dict[str, Any]:
    if ledger is None:
        return existing
    prior = IngestCostLedger.from_dict(existing.get("cost"))
    prior.merge(ledger)
    existing = {**existing, "cost": prior.to_dict()}
    return existing
