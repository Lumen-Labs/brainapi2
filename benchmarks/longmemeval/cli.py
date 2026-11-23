from __future__ import annotations

import argparse
import json
from pathlib import Path

from rich.console import Console
from rich.table import Table

from longmemeval.answer import answer_question
from longmemeval.client import BrainAPIClient
from longmemeval.config import VARIANT_FILES, Settings
from longmemeval.dataset import (
    brain_id_for,
    dataset_stats,
    download_dataset,
    get_question,
    iter_ingest_units,
    load_dataset,
    resolve_questions,
    resolve_variant_settings,
)
from longmemeval.evaluate import evaluate_questions
from longmemeval.ingest import ensure_run_dir, ingest_questions, write_manifest
from longmemeval.metrics import selftest_metrics, tokenize
from longmemeval.prompts import ANSWER_SYSTEM
from longmemeval.provenance import build_provenance
from longmemeval.report import print_report_table, write_report
from degraded import (
    DEFAULT_MAX_DEGRADED_RATE,
    add_max_degraded_rate_argument,
    report_exit_code,
)

console = Console()


def _settings_for_args(args: argparse.Namespace, settings: Settings) -> Settings:
    variant = getattr(args, "variant", None)
    return resolve_variant_settings(settings, variant)


def cmd_download(args: argparse.Namespace, settings: Settings) -> int:
    settings = _settings_for_args(args, settings)
    path = download_dataset(
        settings.dataset_path, settings.dataset_url, variant=settings.variant
    )
    console.print(f"[green]Dataset ready:[/green] {path} (variant={settings.variant})")
    return 0


def cmd_dataset_stats(args: argparse.Namespace, settings: Settings) -> int:
    settings = _settings_for_args(args, settings)
    if not settings.dataset_path.exists():
        download_dataset(
            settings.dataset_path, settings.dataset_url, variant=settings.variant
        )
    stats = dataset_stats(load_dataset(settings.dataset_path, variant=settings.variant))
    table = Table(title=f"LongMemEval ({settings.variant})")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Questions", str(stats["n_questions"]))
    table.add_row("Abstention", str(stats["n_abstention"]))
    sess = stats["sessions_per_question"]
    table.add_row(
        "Sessions/question",
        f"min={sess['min']} max={sess['max']} mean={sess['mean']:.1f}",
    )
    console.print(table)
    type_table = Table(title="By question type")
    type_table.add_column("Type")
    type_table.add_column("N", justify="right")
    for qtype, count in (stats.get("by_type") or {}).items():
        type_table.add_row(qtype, str(count))
    console.print(type_table)
    assert brain_id_for("e47b4ab3aa_abs") == "lmee47b4ab3aaabs"
    console.print("[dim]brain_id_for('e47b4ab3aa_abs') == 'lmee47b4ab3aaabs' OK[/dim]")
    return 0


def cmd_smoke(args: argparse.Namespace, settings: Settings) -> int:
    settings = _settings_for_args(args, settings)
    if not settings.dataset_path.exists():
        download_dataset(
            settings.dataset_path, settings.dataset_url, variant=settings.variant
        )
    questions = resolve_questions(settings, limit=max(1, args.limit))
    entry = questions[0]
    question_id = str(entry["question_id"])
    brain_id = args.brain or brain_id_for(question_id)
    units = iter_ingest_units(entry, limit_sessions=1)
    if not units:
        raise SystemExit("No haystack sessions to smoke-test")
    unit = units[0]
    with BrainAPIClient(settings) as client:
        submitted = client.ingest_text(
            unit["text"],
            brain_id,
            source_timestamp=unit.get("source_timestamp"),
        )
        task_id = (submitted.data or {}).get("task_id")
        console.print(f"Queued ingest task_id={task_id} brain={brain_id}")
        waited = client.wait_for_task(task_id, brain_id, timeout_s=args.timeout)
        console.print(f"Task status: {(waited.data or {}).get('status')}")
        question = str(entry.get("question") or "What do you remember?")
        ctx = client.retrieve_context(question, brain_id)
        text_context = (ctx.data or {}).get("text_context") or ""
        console.print(f"text_context ({len(text_context)} chars): {text_context[:400]}")
        if not text_context.strip():
            console.print("[yellow]Warning: empty text_context[/yellow]")
            return 1
    console.print("[green]Smoke test passed[/green]")
    return 0


def cmd_ingest(args: argparse.Namespace, settings: Settings) -> int:
    settings = _settings_for_args(args, settings)
    if not settings.dataset_path.exists():
        download_dataset(
            settings.dataset_path, settings.dataset_url, variant=settings.variant
        )
    question_types = (
        {t.strip() for t in args.question_type.split(",") if t.strip()}
        if args.question_type
        else None
    )
    questions = resolve_questions(
        settings,
        question_ids=args.question_id or None,
        limit=args.limit,
        question_types=question_types,
    )
    run_id, run_dir = ensure_run_dir(settings, args.run)
    write_manifest(
        run_dir,
        {
            "run_id": run_id,
            "command": "ingest",
            "variant": settings.variant,
            "n_questions": len(questions),
            "question_ids": [q["question_id"] for q in questions],
            "concurrency": args.concurrency,
            "limit": args.limit,
            "limit_sessions": args.limit_sessions,
            "dry_run": args.dry_run,
            "brainapi_url": settings.brainapi_url,
            **build_provenance(settings),
        },
    )
    ingest_questions(
        settings,
        questions,
        run_dir=run_dir,
        concurrency=args.concurrency,
        limit_sessions=args.limit_sessions,
        dry_run=args.dry_run,
        resume=not args.no_resume,
        task_timeout_s=args.timeout,
    )
    console.print(f"[green]Run directory:[/green] {run_dir}")
    return 0


def cmd_answer_once(args: argparse.Namespace, settings: Settings) -> int:
    settings = _settings_for_args(args, settings)
    entry = get_question(
        load_dataset(settings.dataset_path, variant=settings.variant),
        args.question_id,
    )
    brain_id = brain_id_for(str(entry["question_id"]))
    question = args.question or str(entry.get("question") or "")
    with BrainAPIClient(settings) as client:
        retrieved = client.retrieve_context(question, brain_id)
    result = answer_question(
        settings,
        question,
        retrieved.data or {},
        question_date=str(entry.get("question_date") or "") or None,
    )
    console.print(result.answer)
    console.print(
        f"[dim]model={result.model} latency_ms={result.latency_ms:.0f} "
        f"tokens={result.total_tokens}[/dim]"
    )
    return 0


def cmd_selftest_metrics(args: argparse.Namespace, settings: Settings) -> int:
    errors = selftest_metrics()
    if errors:
        for err in errors:
            console.print(f"[red]FAIL[/red] {err}")
        return 1
    console.print("[green]selftest-metrics passed[/green]")
    return 0


def cmd_evaluate(args: argparse.Namespace, settings: Settings) -> int:
    settings = _settings_for_args(args, settings)
    if not settings.dataset_path.exists():
        download_dataset(
            settings.dataset_path, settings.dataset_url, variant=settings.variant
        )
    question_types = (
        {t.strip() for t in args.question_type.split(",") if t.strip()}
        if args.question_type
        else None
    )
    questions = resolve_questions(
        settings,
        question_ids=args.question_id or None,
        limit=args.limit,
        question_types=question_types,
    )
    run_id, run_dir = ensure_run_dir(settings, args.run)
    if settings.judge_shares_answer_family:
        console.print(
            f"[yellow]Warning: judge model ({settings.judge_model}) is the same "
            f"family as the answer model ({settings.answer_model}); judge accuracy "
            "carries self-preference bias.[/yellow]"
        )
    write_manifest(
        run_dir,
        {
            "run_id": run_id,
            "command": "evaluate",
            "variant": settings.variant,
            "n_questions": len(questions),
            "question_ids": [q["question_id"] for q in questions],
            "concurrency": args.concurrency,
            "limit": args.limit,
            "brainapi_url": settings.brainapi_url,
            **build_provenance(settings),
            "historical_limit": args.historical_limit,
            "max_passages": args.max_passages,
            "max_facts": args.max_facts,
            "apply_fact_filter": not args.no_fact_filter,
            "use_ppr": args.use_ppr,
            "sufficiency_retry": args.sufficiency_retry,
        },
    )
    evaluate_questions(
        settings,
        questions,
        run_dir=run_dir,
        concurrency=args.concurrency,
        resume=not args.no_resume,
        historical_limit=args.historical_limit,
        max_passages=args.max_passages,
        max_facts=args.max_facts,
        apply_fact_filter=not args.no_fact_filter,
        use_ppr=args.use_ppr,
        sufficiency_retry=args.sufficiency_retry,
    )
    report = write_report(
        run_dir,
        max_degraded_rate=float(
            getattr(args, "max_degraded_rate", DEFAULT_MAX_DEGRADED_RATE)
        ),
    )
    print_report_table(report)
    console.print(f"[green]Report:[/green] {run_dir / 'report.md'}")
    return report_exit_code(report)


def cmd_report(args: argparse.Namespace, settings: Settings) -> int:
    run_dir = settings.runs_dir / args.run
    if not run_dir.exists():
        raise SystemExit(f"Run directory not found: {run_dir}")
    report = write_report(
        run_dir,
        max_degraded_rate=float(
            getattr(args, "max_degraded_rate", DEFAULT_MAX_DEGRADED_RATE)
        ),
    )
    print_report_table(report)
    console.print(report["run_dir"] + "/report.md")
    if args.json:
        console.print_json(json.dumps(report))
    return report_exit_code(report)


def cmd_prompt_audit(args: argparse.Namespace, settings: Settings) -> int:
    settings = _settings_for_args(args, settings)
    if not settings.dataset_path.exists():
        download_dataset(
            settings.dataset_path, settings.dataset_url, variant=settings.variant
        )
    dataset = load_dataset(settings.dataset_path, variant=settings.variant)
    prompt_tokens = tokenize(ANSWER_SYSTEM)
    n = args.ngram
    prompt_ngrams = {
        tuple(prompt_tokens[i : i + n])
        for i in range(0, max(0, len(prompt_tokens) - n + 1))
    }
    hits: list[tuple[str, str]] = []
    for entry in dataset:
        qid = str(entry.get("question_id"))
        gold_tokens = tokenize(str(entry.get("answer") or ""))
        for i in range(0, max(0, len(gold_tokens) - n + 1)):
            gram = tuple(gold_tokens[i : i + n])
            if gram in prompt_ngrams:
                hits.append((qid, " ".join(gram)))
                break
    if hits:
        console.print(
            f"[red]FAIL[/red] answer prompt shares a {n}-gram with "
            f"{len(hits)} gold answers"
        )
        for qid, gram in hits[:20]:
            console.print(f"  {qid}  {gram!r}")
        return 1
    console.print(
        f"[green]prompt-audit passed[/green] no {n}-gram of the answer prompt "
        f"appears in any gold answer ({len(dataset)} questions)"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m longmemeval",
        description="LongMemEval benchmark harness for BrainAPI",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=None,
        help="Optional path to .env (defaults to benchmarks/.env)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_variant(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--variant",
            choices=sorted(VARIANT_FILES),
            default=None,
            help="Dataset variant (default: s / BENCH_LME_VARIANT)",
        )

    p_download = sub.add_parser("download", help="Download LongMemEval JSON")
    add_variant(p_download)
    p_download.set_defaults(func=cmd_download)

    p_stats = sub.add_parser("dataset-stats", help="Print dataset statistics")
    add_variant(p_stats)
    p_stats.set_defaults(func=cmd_dataset_stats)

    p_smoke = sub.add_parser("smoke", help="Smoke-test ingest + retrieve")
    add_variant(p_smoke)
    p_smoke.add_argument("--brain", default=None)
    p_smoke.add_argument("--limit", type=int, default=1)
    p_smoke.add_argument("--timeout", type=float, default=600.0)
    p_smoke.set_defaults(func=cmd_smoke)

    p_ingest = sub.add_parser("ingest", help="Ingest LongMemEval haystacks")
    add_variant(p_ingest)
    p_ingest.add_argument("--run", default=None, help="Existing or new run id")
    p_ingest.add_argument(
        "--question-id",
        action="append",
        help="Question id (repeatable). Default: all questions.",
    )
    p_ingest.add_argument("--question-type", default=None, help="Comma-separated types")
    p_ingest.add_argument("--limit", type=int, default=None)
    p_ingest.add_argument("--limit-sessions", type=int, default=None)
    p_ingest.add_argument("--concurrency", type=int, default=2)
    p_ingest.add_argument("--dry-run", action="store_true")
    p_ingest.add_argument("--no-resume", action="store_true")
    p_ingest.add_argument("--timeout", type=float, default=900.0)
    p_ingest.set_defaults(func=cmd_ingest)

    p_answer = sub.add_parser("answer-once", help="Retrieve + answer one question")
    add_variant(p_answer)
    p_answer.add_argument("--question-id", required=True)
    p_answer.add_argument("--question", default=None)
    p_answer.set_defaults(func=cmd_answer_once)

    p_self = sub.add_parser("selftest-metrics", help="Run local metrics self-test")
    p_self.set_defaults(func=cmd_selftest_metrics)

    p_eval = sub.add_parser("evaluate", help="Answer and score LongMemEval QA")
    add_variant(p_eval)
    p_eval.add_argument("--run", default=None)
    p_eval.add_argument("--question-id", action="append")
    p_eval.add_argument("--question-type", default=None)
    p_eval.add_argument("--limit", type=int, default=None)
    p_eval.add_argument("--concurrency", type=int, default=2)
    p_eval.add_argument("--no-resume", action="store_true")
    p_eval.add_argument("--historical-limit", type=int, default=10)
    p_eval.add_argument("--max-passages", type=int, default=8)
    p_eval.add_argument("--max-facts", type=int, default=40)
    p_eval.add_argument("--no-fact-filter", action="store_true")
    p_eval.add_argument("--use-ppr", action="store_true")
    p_eval.add_argument("--sufficiency-retry", action="store_true")
    add_max_degraded_rate_argument(p_eval)
    p_eval.set_defaults(func=cmd_evaluate)

    p_report = sub.add_parser("report", help="Rebuild report from a run directory")
    p_report.add_argument("--run", required=True)
    p_report.add_argument("--json", action="store_true")
    add_max_degraded_rate_argument(p_report)
    p_report.set_defaults(func=cmd_report)

    p_audit = sub.add_parser(
        "prompt-audit",
        help="Fail if the answer prompt shares an n-gram with any gold answer",
    )
    add_variant(p_audit)
    p_audit.add_argument("--ngram", type=int, default=3)
    p_audit.set_defaults(func=cmd_prompt_audit)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    settings = Settings.load(args.env_file)
    return args.func(args, settings)
