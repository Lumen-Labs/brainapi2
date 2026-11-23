from __future__ import annotations

import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.progress import Progress

_BENCH_ROOT = Path(__file__).resolve().parent.parent
if str(_BENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(_BENCH_ROOT))

from degraded import degraded_stages, row_is_degraded
from locomo.answer import answer_question
from locomo.client import BrainAPIClient
from locomo.config import Settings
from locomo.dataset import brain_id_for
from locomo.ingest import append_jsonl, load_jsonl
from locomo.judge import judge_answer
from locomo.metrics import evidence_coverage, overlap_scores
from locomo.prompts import (
    attach_image_cues,
    enrich_paths_from_triples,
    flatten_path,
    flatten_triple,
)
from locomo.sota import merge_contexts, plan_gap_fill, plan_selective_second_reflect

console = Console()
_write_lock = threading.Lock()

_GRAPH_CONTEXT_CAP = 60000
_TRIPLES_CAP = 500
_PASSAGE_PREFIX = "[passage] "
_SESSION_RE = re.compile(r"session_(\d+)", re.IGNORECASE)
_DIALOG_RE = re.compile(r"\bD(\d+):", re.IGNORECASE)


@dataclass
class AnswerRecord:
    sample_id: str
    brain_id: str
    qa_index: int
    question: str
    gold: str
    prediction: str
    category: int
    evidence: list[str]
    judge_correct: bool
    judge_reason: str
    judge_raw: str
    f1: float | None
    bleu1: float | None
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
    retrieved_session_ids_graph: list[str] = field(default_factory=list)
    retrieved_session_ids_passages: list[str] = field(default_factory=list)
    context_truncated: dict[str, int] = field(default_factory=dict)
    degraded: list[str] = field(default_factory=list)
    is_adversarial: bool = False
    error: str | None = None
    timestamp: str = ""


def _extract_session_ids(*blobs: Any) -> list[str]:
    found: list[str] = []
    seen: set[str] = set()
    for blob in blobs:
        if isinstance(blob, (list, tuple)):
            text = "\n".join(str(item) for item in blob)
        else:
            text = str(blob or "")
        for match in _SESSION_RE.finditer(text):
            sid = f"session_{match.group(1)}"
            if sid not in seen:
                seen.add(sid)
                found.append(sid)
        for match in _DIALOG_RE.finditer(text):
            sid = f"session_{match.group(1)}"
            if sid not in seen:
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


def extract_context_channels(context: dict[str, Any]) -> dict[str, Any]:
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
    session_ids_graph = _extract_session_ids(graph_context, triples, paths)
    session_ids_passages = _extract_session_ids(source_passages, historical_context)
    truncated: dict[str, int] = {}
    if len(graph_context) > _GRAPH_CONTEXT_CAP:
        truncated["graph_context"] = len(graph_context)
        graph_context = graph_context[:_GRAPH_CONTEXT_CAP]
    if len(triples) > _TRIPLES_CAP:
        truncated["triples"] = len(triples)
        triples = triples[:_TRIPLES_CAP]
    return {
        "graph_context": graph_context,
        "triples": triples,
        "paths": paths,
        "source_passages": source_passages,
        "historical_context": historical_context,
        "retrieved_session_ids": _extract_session_ids(
            session_ids_passages, session_ids_graph
        ),
        "retrieved_session_ids_graph": session_ids_graph,
        "retrieved_session_ids_passages": session_ids_passages,
        "context_truncated": truncated,
        "degraded": degraded_stages(context.get("degraded")),
    }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def qa_gold(qa: dict[str, Any]) -> tuple[str, bool]:
    answer = qa.get("answer")
    if answer is not None and str(answer).strip():
        return str(answer), int(qa.get("category") or 0) == 5
    adversarial = qa.get("adversarial_answer")
    if adversarial is not None:
        return str(adversarial), True
    return "", int(qa.get("category") or 0) == 5


def qa_key(sample_id: str, qa_index: int) -> str:
    return f"{sample_id}::{qa_index}"


def completed_qa_keys(answers_path: Path) -> set[str]:
    done = set()
    for row in load_jsonl(answers_path):
        if row.get("error"):
            continue
        if row.get("prediction") is None:
            continue
        done.add(qa_key(str(row.get("sample_id")), int(row.get("qa_index"))))
    return done


def iter_qa_jobs(
    samples: list[dict[str, Any]],
    *,
    categories: set[int] | None = None,
    skip_adversarial: bool = True,
    limit: int | None = None,
    brain_override: str | None = None,
) -> list[dict[str, Any]]:
    if brain_override and len(samples) > 1:
        raise SystemExit("--brain can only be used with a single --sample")
    jobs: list[dict[str, Any]] = []
    for sample in samples:
        sample_id = str(sample["sample_id"])
        brain_id = brain_override or brain_id_for(sample_id)
        for idx, qa in enumerate(sample.get("qa") or []):
            cat = int(qa.get("category") or 0)
            if skip_adversarial and cat == 5:
                continue
            if categories is not None and cat not in categories:
                continue
            jobs.append(
                {
                    "sample_id": sample_id,
                    "brain_id": brain_id,
                    "qa_index": idx,
                    "qa": qa,
                    "sample": sample,
                }
            )
            if limit is not None and len(jobs) >= limit:
                return jobs
    return jobs


def selftest_records() -> list[str]:
    errors: list[str] = []
    passages = [
        "Conversation between Caroline and Melanie.\nSession id: session_9.\n"
        + ("filler " * 4000),
        "Session id: session_11. Caroline talked about the marathon.",
    ]
    context = {
        "text_context": "\n".join(
            [f"{_PASSAGE_PREFIX}{p}" for p in passages]
            + [
                "Melanie | PUT | keys | IN | slipper (session_3)",
                "[dossier] Melanie: runs charity races (session_3)",
            ]
        ),
        "source_passages": passages,
        "historical_context": ["Session id: session_11. Earlier chunk."],
        "triples": [
            {
                "identified_entity": "Melanie",
                "triple": ["Melanie", "PUT", "keys", "IN", "slipper"],
            }
        ],
        "paths": [
            {
                "hubs": ["event-a", "event-b"],
                "shared_entity_name": "Melanie",
                "legs": [
                    "Actor: Melanie | MADE | Event: put keys | TARGETED | Target: slipper",
                    "Actor: Melanie | MADE | Event: run | TARGETED | Target: race",
                ],
            }
        ],
    }
    channels = extract_context_channels(context)
    record = AnswerRecord(
        sample_id="conv-26",
        brain_id="locomoconv26",
        qa_index=0,
        question="Where did Melanie put the keys?",
        gold="In Melanie's slipper",
        prediction="In Melanie's slipper",
        category=4,
        evidence=["D3:2"],
        judge_correct=True,
        judge_reason="",
        judge_raw="",
        f1=1.0,
        bleu1=1.0,
        retrieve_latency_ms=1.0,
        answer_latency_ms=1.0,
        judge_latency_ms=1.0,
        answer_model="answer-model",
        judge_model="judge-model",
        answer_total_tokens=1,
        judge_total_tokens=1,
        error=None,
        timestamp=_utc_now(),
        **channels,
    )
    row = asdict(record)

    if _PASSAGE_PREFIX in row["graph_context"]:
        errors.append("graph_context must not contain passage lines")
    if "PUT" not in row["graph_context"]:
        errors.append("graph fact line missing from graph_context")
    if not row["triples"]:
        errors.append("triples must be logged as their own field")
    if "slipper" not in row["triples"][0]:
        errors.append("triples must be flattened to readable text")
    if not row["paths"]:
        errors.append("paths must be logged as their own field")
    if "via Melanie" not in row["paths"][0] or "-->" not in row["paths"][0]:
        errors.append("paths must flatten to via-entity leg chains")
    if row["retrieved_session_ids_graph"] != ["session_3"]:
        errors.append(
            f"graph channel session ids wrong: {row['retrieved_session_ids_graph']}"
        )
    if sorted(row["retrieved_session_ids_passages"]) != ["session_11", "session_9"]:
        errors.append(
            "passage channel session ids wrong: "
            f"{row['retrieved_session_ids_passages']}"
        )
    if sorted(row["retrieved_session_ids"]) != [
        "session_11",
        "session_3",
        "session_9",
    ]:
        errors.append("combined session ids should be the union of both channels")
    if evidence_coverage(row, "graph") != "full":
        errors.append("graph-only evidence should score full recall on the graph channel")
    if evidence_coverage(row, "passages") != "none":
        errors.append("graph-only evidence should score no recall on the passage channel")

    long_context = dict(context)
    long_context["text_context"] = "x" * (_GRAPH_CONTEXT_CAP + 10)
    long_context["source_passages"] = []
    capped = extract_context_channels(long_context)
    if len(capped["graph_context"]) != _GRAPH_CONTEXT_CAP:
        errors.append("graph_context should be capped")
    if capped["context_truncated"].get("graph_context") != _GRAPH_CONTEXT_CAP + 10:
        errors.append("truncation must be recorded with the original size")

    gold, adversarial = qa_gold(
        {"category": 5, "adversarial_answer": "No information available"}
    )
    if gold != "No information available" or not adversarial:
        errors.append("adversarial gold must be read from adversarial_answer")
    gold, adversarial = qa_gold({"category": 4, "answer": "7 May 2023"})
    if gold != "7 May 2023" or adversarial:
        errors.append("non-adversarial gold must be read from answer")
    flagged = extract_context_channels({"degraded": ["data.keyword_search"]})
    if flagged.get("degraded") != ["data.keyword_search"]:
        errors.append("degraded retrieve stages must be recorded on answer rows")
    return errors


def evaluate_samples(
    settings: Settings,
    samples: list[dict[str, Any]],
    *,
    run_dir: Path,
    concurrency: int = 2,
    categories: set[int] | None = None,
    skip_adversarial: bool = True,
    limit: int | None = None,
    resume: bool = True,
    historical_limit: int = 10,
    max_passages: int = 8,
    max_facts: int = 40,
    apply_fact_filter: bool = True,
    use_ppr: bool = False,
    sufficiency_retry: bool = False,
    brain_override: str | None = None,
) -> list[AnswerRecord]:
    run_dir.mkdir(parents=True, exist_ok=True)
    answers_path = run_dir / "answers.jsonl"
    already_done = completed_qa_keys(answers_path) if resume else set()

    jobs = iter_qa_jobs(
        samples,
        categories=categories,
        skip_adversarial=skip_adversarial,
        limit=limit,
        brain_override=brain_override,
    )
    jobs = [
        j
        for j in jobs
        if qa_key(j["sample_id"], j["qa_index"]) not in already_done
    ]

    if not jobs:
        console.print("[green]Nothing to evaluate (all QA already scored).[/green]")
        return []

    console.print(
        f"Evaluating {len(jobs)} QA items "
        f"(concurrency={concurrency}, skip_adversarial={skip_adversarial})"
    )

    records: list[AnswerRecord] = []

    def _one(job: dict[str, Any]) -> AnswerRecord:
        qa = job["qa"]
        question = str(qa.get("question") or "")
        gold, is_adversarial = qa_gold(qa)
        category = int(qa.get("category") or 0)
        evidence = list(qa.get("evidence") or [])

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
            context = attach_image_cues(
                context, job.get("sample"), question=question
            )
            retrieve_ms = retrieved.latency_ms
            draft = answer_question(settings, question, context)
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
                    context = attach_image_cues(
                        context, job.get("sample"), question=question
                    )
                    answered = answer_question(settings, question, context)
                else:
                    answered = draft
                # Cap: at most one selective second reflect after primary gap-fill.
                reflect = plan_selective_second_reflect(
                    question, answered.answer, context
                )
                if reflect.needs_retry:
                    answered = answer_question(settings, question, context)
            else:
                answered = draft
            channels = extract_context_channels(context)
            judged = judge_answer(
                settings,
                question,
                gold,
                answered.answer,
                adversarial=is_adversarial,
            )
            f1, bleu1 = overlap_scores(
                answered.answer, gold, adversarial=is_adversarial
            )
            record = AnswerRecord(
                sample_id=job["sample_id"],
                brain_id=job["brain_id"],
                qa_index=job["qa_index"],
                question=question,
                gold=gold,
                prediction=answered.answer,
                category=category,
                evidence=evidence,
                judge_correct=judged.correct,
                judge_reason=judged.reason,
                judge_raw=judged.raw,
                f1=f1,
                bleu1=bleu1,
                retrieve_latency_ms=retrieve_ms,
                answer_latency_ms=answered.latency_ms,
                judge_latency_ms=judged.latency_ms,
                answer_model=answered.model,
                judge_model=judged.model,
                answer_total_tokens=answered.total_tokens,
                judge_total_tokens=judged.total_tokens,
                is_adversarial=is_adversarial,
                error=None,
                timestamp=_utc_now(),
                **channels,
            )
        except Exception as exc:
            record = AnswerRecord(
                sample_id=job["sample_id"],
                brain_id=job["brain_id"],
                qa_index=job["qa_index"],
                question=question,
                gold=gold,
                prediction="",
                category=category,
                evidence=evidence,
                judge_correct=False,
                judge_reason="",
                judge_raw="",
                f1=None,
                bleu1=None,
                retrieve_latency_ms=None,
                answer_latency_ms=None,
                judge_latency_ms=None,
                answer_model=None,
                judge_model=None,
                answer_total_tokens=None,
                judge_total_tokens=None,
                is_adversarial=is_adversarial,
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
