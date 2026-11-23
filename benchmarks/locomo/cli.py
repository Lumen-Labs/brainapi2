from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from locomo.answer import answer_question
from locomo.client import BrainAPIClient
from locomo.config import Settings
from locomo.dataset import (
    brain_id_for,
    dataset_stats,
    download_dataset,
    get_sample,
    load_dataset,
    resolve_samples,
)
from locomo.evaluate import evaluate_samples, selftest_records
from locomo.ingest import ensure_run_dir, ingest_samples, load_jsonl, write_manifest
from locomo.metrics import (
    compare_arms,
    retrieval_arm_summary,
    selftest_metrics,
    tokenize,
)
from locomo.prompts import ANSWER_SYSTEM
from locomo.provenance import build_provenance
from locomo.report import print_comparison, print_report_table, write_report

console = Console()


def _parse_categories(raw: str | None) -> set[int] | None:
    if not raw:
        return None
    return {int(part.strip()) for part in raw.split(",") if part.strip()}


def cmd_download(args: argparse.Namespace, settings: Settings) -> int:
    path = download_dataset(settings.dataset_path, settings.dataset_url)
    console.print(f"[green]Dataset ready:[/green] {path}")
    return 0


def cmd_dataset_stats(args: argparse.Namespace, settings: Settings) -> int:
    if not settings.dataset_path.exists():
        download_dataset(settings.dataset_path, settings.dataset_url)
    rows = dataset_stats(load_dataset(settings.dataset_path))
    table = Table(title="LoCoMo dataset")
    for col in ("sample_id", "brain_id", "speakers", "sessions", "turns", "qa"):
        table.add_column(col)
    for row in rows:
        table.add_row(
            str(row["sample_id"]),
            str(row["brain_id"]),
            f"{row['speaker_a']} & {row['speaker_b']}",
            str(row["sessions"]),
            str(row["turns"]),
            str(row["qa"]),
        )
    console.print(table)
    assert brain_id_for("conv-26") == "locomoconv26"
    console.print("[dim]brain_id_for('conv-26') == 'locomoconv26' OK[/dim]")
    return 0


def cmd_smoke(args: argparse.Namespace, settings: Settings) -> int:
    brain_id = args.brain
    text = (
        "Emily organized the AI Ethics Meetup in London on March 8, 2024. "
        "Caroline attended an LGBTQ support group on 7 May 2023."
    )
    with BrainAPIClient(settings) as client:
        submitted = client.ingest_text(text, brain_id)
        task_id = (submitted.data or {}).get("task_id")
        console.print(f"Queued ingest task_id={task_id}")
        waited = client.wait_for_task(task_id, brain_id, timeout_s=args.timeout)
        console.print(f"Task status: {(waited.data or {}).get('status')}")
        ctx = client.retrieve_context(
            "When did Caroline go to the LGBTQ support group?", brain_id
        )
        text_context = (ctx.data or {}).get("text_context") or ""
        console.print(f"text_context ({len(text_context)} chars): {text_context[:400]}")
        if not text_context.strip():
            console.print("[yellow]Warning: empty text_context[/yellow]")
            return 1
    console.print("[green]Smoke test passed[/green]")
    return 0


def cmd_ingest(args: argparse.Namespace, settings: Settings) -> int:
    sample_ids = args.sample or None
    samples = resolve_samples(settings, sample_ids)
    run_id, run_dir = ensure_run_dir(settings, args.run)
    write_manifest(
        run_dir,
        {
            "run_id": run_id,
            "command": "ingest",
            "samples": [s["sample_id"] for s in samples],
            "granularity": args.granularity,
            "concurrency": args.concurrency,
            "limit_sessions": args.limit_sessions,
            "dry_run": args.dry_run,
            "brainapi_url": settings.brainapi_url,
            "brain_override": args.brain,
            **build_provenance(settings),
        },
    )
    ingest_samples(
        settings,
        samples,
        run_dir=run_dir,
        granularity=args.granularity,
        concurrency=args.concurrency,
        limit_sessions=args.limit_sessions,
        dry_run=args.dry_run,
        resume=not args.no_resume,
        task_timeout_s=args.timeout,
        brain_override=args.brain,
    )
    console.print(f"[green]Run directory:[/green] {run_dir}")
    return 0


def cmd_answer_once(args: argparse.Namespace, settings: Settings) -> int:
    sample = get_sample(load_dataset(settings.dataset_path), args.sample)
    brain_id = brain_id_for(str(sample["sample_id"]))
    question = args.question
    if not question:
        qa = (sample.get("qa") or [None])[0]
        if not qa:
            raise SystemExit("No question provided and sample has no QA")
        question = qa["question"]
        console.print(f"[dim]Using first QA: {question}[/dim]")

    with BrainAPIClient(settings) as client:
        retrieved = client.retrieve_context(question, brain_id)
    result = answer_question(settings, question, retrieved.data or {})
    console.print(result.answer)
    console.print(
        f"[dim]model={result.model} latency_ms={result.latency_ms:.0f} "
        f"tokens={result.total_tokens}[/dim]"
    )
    return 0


def cmd_selftest_metrics(args: argparse.Namespace, settings: Settings) -> int:
    errors = selftest_metrics() + selftest_records()
    if errors:
        for err in errors:
            console.print(f"[red]FAIL[/red] {err}")
        return 1
    console.print("[green]selftest-metrics passed[/green]")
    return 0


def cmd_evaluate(args: argparse.Namespace, settings: Settings) -> int:
    sample_ids = args.sample or None
    samples = resolve_samples(settings, sample_ids)
    run_id, run_dir = ensure_run_dir(settings, args.run)
    categories = _parse_categories(args.categories)
    if settings.judge_shares_answer_family:
        console.print(
            f"[yellow]Warning: judge model ({settings.judge_model}) is the same "
            f"family as the answer model ({settings.answer_model}); judge accuracy "
            "carries self-preference bias. Set BENCH_JUDGE_MODEL (with "
            "BENCH_JUDGE_BASE_URL / BENCH_JUDGE_API_KEY) to a different "
            "family.[/yellow]"
        )
    write_manifest(
        run_dir,
        {
            "run_id": run_id,
            "command": "evaluate",
            "samples": [s["sample_id"] for s in samples],
            "concurrency": args.concurrency,
            "skip_adversarial": not args.include_adversarial,
            "categories": sorted(categories) if categories else None,
            "limit": args.limit,
            "brainapi_url": settings.brainapi_url,
            "brain_override": args.brain,
            **build_provenance(settings),
            "historical_limit": args.historical_limit,
            "max_passages": args.max_passages,
            "max_facts": args.max_facts,
            "apply_fact_filter": not args.no_fact_filter,
            "use_ppr": args.use_ppr,
            "sufficiency_retry": args.sufficiency_retry,
        },
    )
    evaluate_samples(
        settings,
        samples,
        run_dir=run_dir,
        concurrency=args.concurrency,
        categories=categories,
        skip_adversarial=not args.include_adversarial,
        limit=args.limit,
        resume=not args.no_resume,
        historical_limit=args.historical_limit,
        max_passages=args.max_passages,
        max_facts=args.max_facts,
        apply_fact_filter=not args.no_fact_filter,
        use_ppr=args.use_ppr,
        sufficiency_retry=args.sufficiency_retry,
        brain_override=args.brain,
    )
    report = write_report(
        run_dir, include_degraded=bool(getattr(args, "include_degraded", False))
    )
    print_report_table(report)
    console.print(f"[green]Report:[/green] {run_dir / 'report.md'}")
    return 0


def cmd_report(args: argparse.Namespace, settings: Settings) -> int:
    run_dir = settings.runs_dir / args.run
    if not run_dir.exists():
        raise SystemExit(f"Run directory not found: {run_dir}")
    report = write_report(
        run_dir, include_degraded=bool(getattr(args, "include_degraded", False))
    )
    print_report_table(report)
    console.print(report["run_dir"] + "/report.md")
    if args.json:
        console.print_json(json.dumps(report))
    return 0


def _load_run_answers(settings: Settings, run_id: str) -> list[dict[str, Any]]:
    run_dir = settings.runs_dir / run_id
    answers_path = run_dir / "answers.jsonl"
    if not answers_path.exists():
        raise SystemExit(f"No answers.jsonl in run: {run_dir}")
    rows = load_jsonl(answers_path)
    latest: dict[tuple[str, int], dict[str, Any]] = {}
    for row in rows:
        if row.get("error") or row.get("degraded"):
            continue
        latest[(str(row.get("sample_id")), int(row.get("qa_index") or 0))] = row
    return list(latest.values())


def cmd_compare(args: argparse.Namespace, settings: Settings) -> int:
    baseline = [_load_run_answers(settings, run) for run in args.baseline]
    candidate = [_load_run_answers(settings, run) for run in args.candidate]
    comparison = compare_arms(
        baseline, candidate, skip_adversarial=not args.include_adversarial
    )
    retrieval = {
        "baseline": {"labels": args.baseline, **retrieval_arm_summary(baseline)},
        "candidate": {"labels": args.candidate, **retrieval_arm_summary(candidate)},
    }
    print_comparison(
        comparison,
        " + ".join(args.baseline),
        " + ".join(args.candidate),
        retrieval,
    )
    if args.json:
        console.print_json(
            json.dumps({"comparison": comparison, "retrieval": retrieval})
        )
    return 0


def cmd_prompt_audit(args: argparse.Namespace, settings: Settings) -> int:
    dataset = load_dataset(settings.dataset_path)
    prompt_tokens = tokenize(ANSWER_SYSTEM)
    n = args.ngram
    prompt_ngrams = {
        tuple(prompt_tokens[i : i + n])
        for i in range(0, max(0, len(prompt_tokens) - n + 1))
    }
    hits: list[tuple[str, int, str]] = []
    for sample in dataset:
        sample_id = str(sample.get("sample_id"))
        for idx, qa in enumerate(sample.get("qa") or []):
            gold = qa.get("answer")
            if gold is None:
                gold = qa.get("adversarial_answer")
            gold_tokens = tokenize(str(gold or ""))
            for i in range(0, max(0, len(gold_tokens) - n + 1)):
                gram = tuple(gold_tokens[i : i + n])
                if gram in prompt_ngrams:
                    hits.append((sample_id, idx, " ".join(gram)))
                    break
    if hits:
        console.print(
            f"[red]FAIL[/red] answer prompt shares a {n}-gram with "
            f"{len(hits)} gold answers"
        )
        for sample_id, idx, gram in hits[:20]:
            console.print(f"  {sample_id}::{idx}  {gram!r}")
        return 1
    console.print(
        f"[green]prompt-audit passed[/green] no {n}-gram of the answer prompt "
        f"appears in any gold answer ({len(dataset)} conversations)"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m locomo",
        description="LoCoMo benchmark harness for BrainAPI",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=None,
        help="Optional path to .env (defaults to benchmarks/.env)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_download = sub.add_parser("download", help="Download locomo10.json")
    p_download.set_defaults(func=cmd_download)

    p_stats = sub.add_parser("dataset-stats", help="Print dataset statistics")
    p_stats.set_defaults(func=cmd_dataset_stats)

    p_smoke = sub.add_parser("smoke", help="Smoke-test ingest + retrieve")
    p_smoke.add_argument("--brain", default="locomosmoke")
    p_smoke.add_argument("--timeout", type=float, default=600.0)
    p_smoke.set_defaults(func=cmd_smoke)

    p_ingest = sub.add_parser("ingest", help="Ingest LoCoMo conversations")
    p_ingest.add_argument(
        "--sample",
        action="append",
        help="Sample id (repeatable). Default: all samples.",
    )
    p_ingest.add_argument("--run", default=None, help="Existing or new run id")
    p_ingest.add_argument(
        "--brain",
        default=None,
        help="Override the derived brain id (single --sample only)",
    )
    p_ingest.add_argument(
        "--granularity", choices=("session", "turn"), default="session"
    )
    p_ingest.add_argument("--concurrency", type=int, default=2)
    p_ingest.add_argument("--limit-sessions", type=int, default=None)
    p_ingest.add_argument("--dry-run", action="store_true")
    p_ingest.add_argument("--no-resume", action="store_true")
    p_ingest.add_argument("--timeout", type=float, default=900.0)
    p_ingest.set_defaults(func=cmd_ingest)

    p_answer = sub.add_parser("answer-once", help="Retrieve + answer one question")
    p_answer.add_argument("--sample", default="conv-26")
    p_answer.add_argument("--question", default=None)
    p_answer.set_defaults(func=cmd_answer_once)

    p_self = sub.add_parser("selftest-metrics", help="Run local metrics self-test")
    p_self.set_defaults(func=cmd_selftest_metrics)

    p_eval = sub.add_parser("evaluate", help="Answer and score LoCoMo QA")
    p_eval.add_argument("--sample", action="append")
    p_eval.add_argument("--run", default=None)
    p_eval.add_argument(
        "--brain",
        default=None,
        help="Override the derived brain id (single --sample only)",
    )
    p_eval.add_argument("--concurrency", type=int, default=2)
    p_eval.add_argument("--categories", default=None, help="e.g. 1,2,4")
    p_eval.add_argument("--limit", type=int, default=None)
    p_eval.add_argument(
        "--include-adversarial",
        action="store_true",
        help="Include category 5 questions",
    )
    p_eval.add_argument("--no-resume", action="store_true")
    p_eval.add_argument("--historical-limit", type=int, default=10)
    p_eval.add_argument("--max-passages", type=int, default=8)
    p_eval.add_argument("--max-facts", type=int, default=40)
    p_eval.add_argument("--no-fact-filter", action="store_true")
    p_eval.add_argument("--use-ppr", action="store_true")
    p_eval.add_argument("--sufficiency-retry", action="store_true")
    p_eval.add_argument(
        "--include-degraded",
        action="store_true",
        help="Include /retrieve/context degraded rows in accuracy scores",
    )
    p_eval.set_defaults(func=cmd_evaluate)

    p_report = sub.add_parser("report", help="Rebuild report from a run directory")
    p_report.add_argument("--run", required=True)
    p_report.add_argument("--json", action="store_true")
    p_report.add_argument(
        "--include-degraded",
        action="store_true",
        help="Include /retrieve/context degraded rows in accuracy scores",
    )
    p_report.set_defaults(func=cmd_report)

    p_compare = sub.add_parser(
        "compare", help="Paired McNemar comparison between two run arms"
    )
    p_compare.add_argument("--baseline", action="append", required=True)
    p_compare.add_argument("--candidate", action="append", required=True)
    p_compare.add_argument("--include-adversarial", action="store_true")
    p_compare.add_argument("--json", action="store_true")
    p_compare.set_defaults(func=cmd_compare)

    p_audit = sub.add_parser(
        "prompt-audit",
        help="Fail if the answer prompt shares an n-gram with any gold answer",
    )
    p_audit.add_argument("--ngram", type=int, default=3)
    p_audit.set_defaults(func=cmd_prompt_audit)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    settings = Settings.load(args.env_file)
    return args.func(args, settings)
