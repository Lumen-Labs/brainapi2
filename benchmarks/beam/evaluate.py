from __future__ import annotations

import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.progress import Progress
from concurrent.futures import ThreadPoolExecutor, as_completed

_BENCH_ROOT = Path(__file__).resolve().parent.parent
if str(_BENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(_BENCH_ROOT))

from beam.answer import answer_question
from beam.client import BrainAPIClient
from beam.config import ABILITY_NAMES, Settings
from beam.dataset import brain_id_for, iter_probing_jobs
from beam.ingest import append_jsonl, load_jsonl
from beam.judge import judge_rubric_response
from beam.sota import merge_contexts, ordering_aspect_queries, plan_gap_fill
from degraded import degraded_stages, row_is_degraded

console = Console()

DEFAULT_HISTORICAL_LIMIT = 16
DEFAULT_MAX_PASSAGES = 16
DEFAULT_MAX_FACTS = 50


@dataclass
class AnswerRecord:
    sample_id: str
    size: str
    brain_id: str
    ability: str
    qa_index: int
    question: str
    prediction: str
    llm_judge_score: float | None
    tau_norm: float | None
    event_ordering_f1: float | None
    rubric_results: list[dict[str, Any]] = field(default_factory=list)
    retrieve_latency_ms: float | None = None
    answer_latency_ms: float | None = None
    judge_latency_ms: float | None = None
    answer_model: str | None = None
    judge_model: str | None = None
    answer_total_tokens: int | None = None
    judge_total_tokens: int | None = None
    sc_samples_used: int | None = None
    gap_fill_used: bool = False
    degraded: list[str] = field(default_factory=list)
    error: str | None = None
    timestamp: str = ""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def qa_key(sample_id: str, ability: str, qa_index: int) -> str:
    return f"{sample_id}::{ability}::{qa_index}"


def completed_qa_keys(answers_path: Path) -> set[str]:
    done = set()
    for row in load_jsonl(answers_path):
        if row.get("error"):
            continue
        if row.get("prediction") is None:
            continue
        done.add(
            qa_key(
                str(row.get("sample_id")),
                str(row.get("ability")),
                int(row.get("qa_index") or 0),
            )
        )
    return done


def evaluate_samples(
    settings: Settings,
    samples: list[dict[str, Any]],
    *,
    run_dir: Path,
    concurrency: int = 2,
    abilities: set[str] | None = None,
    limit: int | None = None,
    resume: bool = True,
    brain_override: str | None = None,
    historical_limit: int = DEFAULT_HISTORICAL_LIMIT,
    max_passages: int = DEFAULT_MAX_PASSAGES,
    max_facts: int = DEFAULT_MAX_FACTS,
    apply_fact_filter: bool = True,
    use_ppr: bool = True,
    sufficiency_retry: bool = False,
) -> list[AnswerRecord]:
    if brain_override and len(samples) > 1:
        raise SystemExit("--brain can only be used with a single --sample")
    if abilities:
        unknown = abilities - set(ABILITY_NAMES)
        if unknown:
            raise SystemExit(f"Unknown abilities: {sorted(unknown)}")

    run_dir.mkdir(parents=True, exist_ok=True)
    answers_path = run_dir / "answers.jsonl"
    already_done = completed_qa_keys(answers_path) if resume else set()

    jobs: list[dict[str, Any]] = []
    for sample in samples:
        sample_id = str(sample["sample_id"])
        size = str(sample["size"])
        conversation_id = str(sample["conversation_id"])
        brain_id = brain_override or brain_id_for(size, conversation_id)
        for job in iter_probing_jobs(sample, abilities=abilities):
            key = qa_key(sample_id, job["ability"], job["qa_index"])
            if key in already_done:
                continue
            jobs.append(
                {
                    "sample_id": sample_id,
                    "size": size,
                    "brain_id": brain_id,
                    **job,
                }
            )
            if limit is not None and len(jobs) >= limit:
                break
        if limit is not None and len(jobs) >= limit:
            break

    if not jobs:
        console.print("[green]Nothing to evaluate (all questions already scored).[/green]")
        return []

    console.print(
        f"Evaluating {len(jobs)} probing questions "
        f"(concurrency={concurrency}, profile={settings.bench_profile}, "
        f"sc={settings.sc_samples}, gap_fill={settings.gap_fill})"
    )
    records: list[AnswerRecord] = []

    def _one(job: dict[str, Any]) -> AnswerRecord:
        gap_fill_used = False
        try:
            eo_hist = historical_limit
            eo_passages = max_passages
            eo_facts = max_facts
            # Mild EO budget bump only — large fan-out exhausts the PG pool on 1M brains.
            if job["ability"] == "event_ordering":
                eo_hist = max(historical_limit, 14)
                eo_passages = max(max_passages, 14)
                eo_facts = max(max_facts, 40)
            with BrainAPIClient(settings) as client:
                retrieved = client.retrieve_context(
                    job["question"],
                    job["brain_id"],
                    historical_limit=eo_hist,
                    max_passages=eo_passages,
                    max_facts=eo_facts,
                    apply_fact_filter=apply_fact_filter,
                    use_ppr=use_ppr,
                    sufficiency_retry=sufficiency_retry,
                )
                context = retrieved.data or {}
                retrieve_ms = retrieved.latency_ms
                # Skip harness multi-retrieve on dense 1M brains — product already
                # fans out ordering variants; extra calls exhaust the PG pool (HTTP 500).
                if job["ability"] == "event_ordering" and str(job.get("size") or "") != "1M":
                    for aspect_q in ordering_aspect_queries(job["question"])[:1]:
                        extra = client.retrieve_context(
                            aspect_q,
                            job["brain_id"],
                            historical_limit=min(eo_hist, 10),
                            max_passages=min(eo_passages, 10),
                            max_facts=min(eo_facts, 30),
                            apply_fact_filter=apply_fact_filter,
                            use_ppr=use_ppr,
                            sufficiency_retry=False,
                        )
                        retrieve_ms = (retrieve_ms or 0) + (extra.latency_ms or 0)
                        context = merge_contexts(context, extra.data or {})
            draft = answer_question(
                settings,
                job["question"],
                context,
                ability=job["ability"],
            )
            answered = draft
            if settings.gap_fill:
                plan = plan_gap_fill(job["question"], draft.answer, context)
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
                        settings,
                        job["question"],
                        context,
                        ability=job["ability"],
                    )
                    gap_fill_used = True
            judged = judge_rubric_response(
                settings,
                ability=job["ability"],
                question=job["question"],
                prediction=answered.answer,
                rubric=[str(x) for x in (job.get("rubric") or [])],
            )
            record = AnswerRecord(
                sample_id=job["sample_id"],
                size=job["size"],
                brain_id=job["brain_id"],
                ability=job["ability"],
                qa_index=job["qa_index"],
                question=job["question"],
                prediction=answered.answer,
                llm_judge_score=judged.llm_judge_score,
                tau_norm=judged.tau_norm,
                event_ordering_f1=judged.event_ordering_f1,
                rubric_results=[
                    {
                        "score": item.score,
                        "reason": item.reason,
                    }
                    for item in judged.rubric_results
                ],
                retrieve_latency_ms=retrieve_ms,
                answer_latency_ms=answered.latency_ms,
                judge_latency_ms=judged.latency_ms,
                answer_model=answered.model,
                judge_model=judged.model,
                answer_total_tokens=answered.total_tokens,
                judge_total_tokens=judged.total_tokens,
                sc_samples_used=answered.sc_samples_used,
                gap_fill_used=gap_fill_used,
                degraded=degraded_stages(context.get("degraded")),
                timestamp=_utc_now(),
            )
        except Exception as exc:
            record = AnswerRecord(
                sample_id=job["sample_id"],
                size=job["size"],
                brain_id=job["brain_id"],
                ability=job["ability"],
                qa_index=job["qa_index"],
                question=job["question"],
                prediction="",
                llm_judge_score=None,
                tau_norm=None,
                event_ordering_f1=None,
                error=str(exc),
                timestamp=_utc_now(),
            )
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
