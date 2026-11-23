from __future__ import annotations

import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.progress import Progress

from longmemeval.answer import answer_question
from longmemeval.client import BrainAPIClient
from longmemeval.config import Settings
from longmemeval.dataset import brain_id_for, is_abstention
from longmemeval.ingest import append_jsonl, load_jsonl
from longmemeval.judge import judge_answer
from longmemeval.prompts import enrich_paths_from_triples, flatten_path, flatten_triple
from longmemeval.sota import merge_contexts, plan_gap_fill

console = Console()
_write_lock = threading.Lock()

_GRAPH_CONTEXT_CAP = 60000
_TRIPLES_CAP = 500
_PASSAGE_PREFIX = "[passage] "
_SESSION_ID_RE = re.compile(r"Session id:\s*([^\s.]+)", re.IGNORECASE)


@dataclass
class AnswerRecord:
    question_id: str
    brain_id: str
    question: str
    gold: str
    prediction: str
    question_type: str
    question_date: str
    answer_session_ids: list[str]
    judge_correct: bool
    judge_reason: str
    judge_raw: str
    retrieve_latency_ms: float | None
    answer_latency_ms: float | None
    judge_latency_ms: float | None
    answer_model: str | None
    judge_model: str | None
    answer_total_tokens: int | None
    judge_total_tokens: int | None
    graph_context: str
    triples: list[str] = field(default_factory=list)
    paths: list[str] = field(default_factory=list)
    source_passages: list[str] = field(default_factory=list)
    historical_context: list[str] = field(default_factory=list)
    retrieved_session_ids: list[str] = field(default_factory=list)
    context_truncated: dict[str, int] = field(default_factory=dict)
    degraded: list[str] = field(default_factory=list)
    is_abstention: bool = False
    error: str | None = None
    timestamp: str = ""


def degraded_stages(value: Any) -> list[str]:
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if str(item).strip()]
    return []


def row_is_degraded(row: dict[str, Any] | None) -> bool:
    return bool(degraded_stages((row or {}).get("degraded")))


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _extract_session_ids(*blobs: Any, known_ids: list[str] | None = None) -> list[str]:
    found: list[str] = []
    seen: set[str] = set()
    texts: list[str] = []
    for blob in blobs:
        if isinstance(blob, (list, tuple)):
            texts.append("\n".join(str(item) for item in blob))
        else:
            texts.append(str(blob or ""))
    joined = "\n".join(texts)
    for match in _SESSION_ID_RE.finditer(joined):
        sid = match.group(1).rstrip(".")
        if sid and sid not in seen:
            seen.add(sid)
            found.append(sid)
    if known_ids:
        lower_joined = joined.lower()
        for sid in known_ids:
            if sid and sid not in seen and sid.lower() in lower_joined:
                seen.add(sid)
                found.append(sid)
    return found


def _graph_channel(text_context: str, source_passages: list[str]) -> str:
    passage_block = "\n".join(f"{_PASSAGE_PREFIX}{p}" for p in source_passages)
    if passage_block and text_context.startswith(passage_block):
        return text_context[len(passage_block) :].lstrip("\n")
    return "\n".join(
        line
        for line in text_context.splitlines()
        if not line.startswith(_PASSAGE_PREFIX)
    )


def extract_context_channels(
    context: dict[str, Any],
    *,
    known_session_ids: list[str] | None = None,
) -> dict[str, Any]:
    context = enrich_paths_from_triples(context)
    source_passages = [str(p) for p in (context.get("source_passages") or []) if p]
    historical_context = [
        str(h) for h in (context.get("historical_context") or []) if h
    ]
    triples = [flatten_triple(t) for t in (context.get("triples") or []) if t]
    paths = [flatten_path(p) for p in (context.get("paths") or []) if p]
    graph_context = _graph_channel(
        str(context.get("text_context") or ""), source_passages
    )
    truncated: dict[str, int] = {}
    if len(graph_context) > _GRAPH_CONTEXT_CAP:
        truncated["graph_context"] = len(graph_context)
        graph_context = graph_context[:_GRAPH_CONTEXT_CAP]
    if len(triples) > _TRIPLES_CAP:
        truncated["triples"] = len(triples)
        triples = triples[:_TRIPLES_CAP]
    session_ids = _extract_session_ids(
        graph_context,
        triples,
        paths,
        source_passages,
        historical_context,
        known_ids=known_session_ids,
    )
    return {
        "graph_context": graph_context,
        "triples": triples,
        "paths": paths,
        "source_passages": source_passages,
        "historical_context": historical_context,
        "retrieved_session_ids": session_ids,
        "context_truncated": truncated,
        "degraded": degraded_stages(context.get("degraded")),
    }


def completed_question_ids(answers_path: Path) -> set[str]:
    done = set()
    for row in load_jsonl(answers_path):
        if row.get("error"):
            continue
        qid = row.get("question_id")
        if qid:
            done.add(str(qid))
    return done


def evaluate_questions(
    settings: Settings,
    questions: list[dict[str, Any]],
    *,
    run_dir: Path,
    concurrency: int = 2,
    resume: bool = True,
    historical_limit: int = 10,
    max_passages: int = 8,
    max_facts: int = 40,
    apply_fact_filter: bool = True,
    use_ppr: bool = False,
    sufficiency_retry: bool = False,
) -> list[AnswerRecord]:
    run_dir.mkdir(parents=True, exist_ok=True)
    answers_path = run_dir / "answers.jsonl"
    already_done = completed_question_ids(answers_path) if resume else set()

    jobs = [
        {
            "entry": entry,
            "question_id": str(entry["question_id"]),
            "brain_id": brain_id_for(str(entry["question_id"])),
        }
        for entry in questions
        if str(entry["question_id"]) not in already_done
    ]

    if not jobs:
        console.print("[green]Nothing to evaluate (all questions already scored).[/green]")
        return []

    console.print(f"Evaluating {len(jobs)} questions (concurrency={concurrency})")

    records: list[AnswerRecord] = []

    def _one(job: dict[str, Any]) -> AnswerRecord:
        entry = job["entry"]
        question_id = job["question_id"]
        question = str(entry.get("question") or "")
        gold = str(entry.get("answer") or "")
        question_type = str(entry.get("question_type") or "")
        question_date = str(entry.get("question_date") or "")
        answer_session_ids = [str(s) for s in (entry.get("answer_session_ids") or [])]
        abstention = is_abstention(question_id)
        known_ids = [str(s) for s in (entry.get("haystack_session_ids") or [])]

        try:
            with BrainAPIClient(settings) as client:
                retrieved = client.retrieve_context(
                    question,
                    job["brain_id"],
                    historical_limit=historical_limit,
                    max_passages=max_passages,
                    max_facts=max_facts,
                    apply_fact_filter=apply_fact_filter,
                    use_ppr=use_ppr,
                    sufficiency_retry=sufficiency_retry,
                )
            context = retrieved.data or {}
            retrieve_ms = retrieved.latency_ms
            draft = answer_question(
                settings, question, context, question_date=question_date
            )
            if settings.gap_fill:
                plan = plan_gap_fill(question, draft.answer, context)
                if plan.needs_retry:
                    with BrainAPIClient(settings) as client:
                        second = client.retrieve_context(
                            plan.reformulated_query,
                            job["brain_id"],
                            historical_limit=historical_limit,
                            max_passages=max_passages,
                            max_facts=max_facts,
                            apply_fact_filter=apply_fact_filter,
                            use_ppr=use_ppr,
                            sufficiency_retry=sufficiency_retry,
                        )
                    retrieve_ms = (retrieve_ms or 0) + (second.latency_ms or 0)
                    context = merge_contexts(context, second.data or {})
                    answered = answer_question(
                        settings, question, context, question_date=question_date
                    )
                else:
                    answered = draft
            else:
                answered = draft
            channels = extract_context_channels(
                context, known_session_ids=known_ids
            )
            judged = judge_answer(
                settings,
                question,
                gold,
                answered.answer,
                question_type=question_type,
                abstention=abstention,
            )
            record = AnswerRecord(
                question_id=question_id,
                brain_id=job["brain_id"],
                question=question,
                gold=gold,
                prediction=answered.answer,
                question_type=question_type,
                question_date=question_date,
                answer_session_ids=answer_session_ids,
                judge_correct=judged.correct,
                judge_reason=judged.reason,
                judge_raw=judged.raw,
                retrieve_latency_ms=retrieve_ms,
                answer_latency_ms=answered.latency_ms,
                judge_latency_ms=judged.latency_ms,
                answer_model=answered.model,
                judge_model=judged.model,
                answer_total_tokens=answered.total_tokens,
                judge_total_tokens=judged.total_tokens,
                is_abstention=abstention,
                error=None,
                timestamp=_utc_now(),
                **channels,
            )
        except Exception as exc:
            record = AnswerRecord(
                question_id=question_id,
                brain_id=job["brain_id"],
                question=question,
                gold=gold,
                prediction="",
                question_type=question_type,
                question_date=question_date,
                answer_session_ids=answer_session_ids,
                judge_correct=False,
                judge_reason="",
                judge_raw="",
                retrieve_latency_ms=None,
                answer_latency_ms=None,
                judge_latency_ms=None,
                answer_model=None,
                judge_model=None,
                answer_total_tokens=None,
                judge_total_tokens=None,
                is_abstention=abstention,
                error=str(exc),
                timestamp=_utc_now(),
                **extract_context_channels({}),
            )

        with _write_lock:
            append_jsonl(answers_path, asdict(record))
        return record

    with Progress(console=console) as progress:
        task = progress.add_task("evaluate", total=len(jobs))
        with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
            futures = [pool.submit(_one, job) for job in jobs]
            for future in as_completed(futures):
                records.append(future.result())
                progress.advance(task)

    ok = sum(1 for r in records if not r.error)
    failed = sum(1 for r in records if r.error)
    console.print(f"Evaluate done: ok={ok} failed={failed} -> {answers_path}")
    return records
