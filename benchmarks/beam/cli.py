from __future__ import annotations

import argparse
from pathlib import Path

from rich.console import Console
from rich.table import Table

from beam.client import BrainAPIClient
from beam.config import ABILITY_NAMES, CHAT_SIZES, Settings
from beam.dataset import (
    brain_id_for,
    dataset_stats,
    download_and_normalize,
    iter_ingest_units,
    iter_probing_jobs,
    list_local_samples,
    load_sample,
    resolve_samples,
)
from beam.evaluate import (
    DEFAULT_HISTORICAL_LIMIT,
    DEFAULT_MAX_FACTS,
    DEFAULT_MAX_PASSAGES,
    evaluate_samples,
)
from beam.ingest import ensure_run_dir, ingest_samples, write_manifest
from beam.judge import parse_json_response
from beam.metrics import selftest_metrics
from beam.prompts import build_rubric_judge_prompt
from beam.provenance import build_provenance
from beam.report import print_report_table, write_report
from degraded import (
    DEFAULT_MAX_DEGRADED_RATE,
    add_max_degraded_rate_argument,
    report_exit_code,
)

console = Console()


def _parse_abilities(raw: str | None) -> set[str] | None:
    if not raw:
        return None
    return {part.strip() for part in raw.split(",") if part.strip()}


def cmd_download(args: argparse.Namespace, settings: Settings) -> int:
    sizes = [args.size] if args.size else list(CHAT_SIZES)
    path = download_and_normalize(settings, sizes=sizes, force=args.force)
    console.print(f"[green]Dataset ready:[/green] {path}")
    for size in sizes:
        n = len(list_local_samples(settings, size=size))
        console.print(f"  {size}: {n} conversations")
    return 0


def cmd_dataset_stats(args: argparse.Namespace, settings: Settings) -> int:
    rows = dataset_stats(settings)
    if not rows:
        console.print(
            "[yellow]No local BEAM data. Run: python -m beam download[/yellow]"
        )
        return 1
    table = Table(title="BEAM dataset")
    for col in ("sample_id", "brain_id", "size", "batches", "turns", "questions"):
        table.add_column(col)
    for row in rows:
        if args.size and row["size"] != args.size:
            continue
        table.add_row(
            str(row["sample_id"]),
            str(row["brain_id"]),
            str(row["size"]),
            str(row["batches"]),
            str(row["turns"]),
            str(row["questions"]),
        )
    console.print(table)
    assert brain_id_for("100K", "1") == "beam100k1"
    assert brain_id_for("10M", "1") == "beam10m1"
    console.print("[dim]brain_id_for('100K', '1') == 'beam100k1' OK[/dim]")
    console.print("[dim]brain_id_for('10M', '1') == 'beam10m1' OK[/dim]")
    return 0


def cmd_smoke(args: argparse.Namespace, settings: Settings) -> int:
    sample_id = args.sample if "/" in args.sample else f"{args.size}/{args.sample}"
    try:
        sample = load_sample(settings, sample_id)
    except FileNotFoundError:
        console.print(f"[yellow]Sample missing; downloading {args.size}…[/yellow]")
        download_and_normalize(settings, sizes=[args.size])
        sample = load_sample(settings, sample_id)

    brain_id = args.brain or str(sample["brain_id"])
    units = iter_ingest_units(sample, limit_turns=max(1, args.limit_turns))
    if not units:
        console.print("[red]No turns to ingest[/red]")
        return 1

    with BrainAPIClient(settings) as client:
        for unit in units:
            submitted = client.ingest_text(
                unit["text"],
                brain_id,
                source_timestamp=unit.get("source_timestamp"),
            )
            task_id = (submitted.data or {}).get("task_id")
            console.print(f"Queued ingest task_id={task_id} unit={unit['unit_id']}")
            waited = client.wait_for_task(task_id, brain_id, timeout_s=args.timeout)
            console.print(f"Task status: {(waited.data or {}).get('status')}")

        jobs = iter_probing_jobs(sample, limit=1)
        question = jobs[0]["question"] if jobs else "What was discussed?"
        ctx = client.retrieve_context(question, brain_id)
        text_context = (ctx.data or {}).get("text_context") or ""
        console.print(
            f"text_context ({len(text_context)} chars): {text_context[:400]}"
        )
        if not text_context.strip():
            console.print("[yellow]Warning: empty text_context[/yellow]")
            return 1
    console.print("[green]Smoke test passed[/green]")
    return 0


def cmd_ingest(args: argparse.Namespace, settings: Settings) -> int:
    samples = resolve_samples(
        settings, size=args.size, sample_ids=args.sample or None
    )
    run_id, run_dir = ensure_run_dir(settings, args.run)
    write_manifest(
        run_dir,
        {
            "run_id": run_id,
            "command": "ingest",
            "suite": "beam",
            "size": args.size,
            "samples": [s["sample_id"] for s in samples],
            "concurrency": args.concurrency,
            "limit_turns": args.limit_turns,
            "dry_run": args.dry_run,
            "brainapi_url": settings.brainapi_url,
            "brain_override": args.brain,
            **build_provenance(settings, size=args.size),
        },
    )
    ingest_samples(
        settings,
        samples,
        run_dir=run_dir,
        concurrency=args.concurrency,
        limit_turns=args.limit_turns,
        dry_run=args.dry_run,
        resume=not args.no_resume,
        task_timeout_s=args.timeout,
        brain_override=args.brain,
    )
    console.print(f"[green]Run directory:[/green] {run_dir}")
    return 0


def cmd_evaluate(args: argparse.Namespace, settings: Settings) -> int:
    samples = resolve_samples(
        settings, size=args.size, sample_ids=args.sample or None
    )
    run_id, run_dir = ensure_run_dir(settings, args.run)
    abilities = _parse_abilities(args.abilities)
    if getattr(args, "profile", None):
        from beam.sota import profile_defaults

        defaults = profile_defaults(args.profile)
        settings.bench_profile = str(defaults["bench_profile"])
        settings.sc_samples = max(1, int(defaults["sc_samples"]))
        settings.sc_temperature = float(defaults["sc_temperature"])
        settings.gap_fill = bool(defaults["gap_fill"])
    if settings.judge_shares_answer_family:
        console.print(
            f"[yellow]Warning: judge model ({settings.judge_model}) is the same "
            f"family as the answer model ({settings.answer_model}); judge scores "
            "may carry self-preference bias.[/yellow]"
        )
    use_ppr = not bool(getattr(args, "no_ppr", False))
    write_manifest(
        run_dir,
        {
            "run_id": run_id,
            "command": "evaluate",
            "suite": "beam",
            "size": args.size,
            "samples": [s["sample_id"] for s in samples],
            "concurrency": args.concurrency,
            "abilities": sorted(abilities) if abilities else None,
            "limit": args.limit,
            "brainapi_url": settings.brainapi_url,
            "brain_override": args.brain,
            "bench_profile": settings.bench_profile,
            "sc_samples": settings.sc_samples,
            "sc_temperature": settings.sc_temperature,
            "gap_fill": settings.gap_fill,
            **build_provenance(settings, size=args.size),
            "historical_limit": args.historical_limit,
            "max_passages": args.max_passages,
            "max_facts": args.max_facts,
            "apply_fact_filter": not args.no_fact_filter,
            "use_ppr": use_ppr,
            "sufficiency_retry": args.sufficiency_retry,
        },
    )
    evaluate_samples(
        settings,
        samples,
        run_dir=run_dir,
        concurrency=args.concurrency,
        abilities=abilities,
        limit=args.limit,
        resume=not args.no_resume,
        brain_override=args.brain,
        historical_limit=args.historical_limit,
        max_passages=args.max_passages,
        max_facts=args.max_facts,
        apply_fact_filter=not args.no_fact_filter,
        use_ppr=use_ppr,
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
    console.print(f"[green]Report:[/green] {run_dir / 'report.md'}")
    return report_exit_code(report)


def cmd_selftest(args: argparse.Namespace, settings: Settings) -> int:
    errors = selftest_metrics()
    if brain_id_for("100K", "1") != "beam100k1":
        errors.append("brain_id_for('100K','1') != 'beam100k1'")
    if brain_id_for("1M", "12") != "beam1m12":
        errors.append("brain_id_for('1M','12') != 'beam1m12'")
    if brain_id_for("10M", "1") != "beam10m1":
        errors.append("brain_id_for('10M','1') != 'beam10m1'")

    prompt = build_rubric_judge_prompt(
        "When does my sprint end?",
        "LLM response should state: March 29",
        "The sprint ends on March 29.",
    )
    if "<question>" in prompt or "<rubric_item>" in prompt or "<llm_response>" in prompt:
        errors.append("rubric judge prompt still contains placeholders")

    payload = parse_json_response('{"score": 0.5, "reason": "partial"}')
    if float(payload["score"]) != 0.5:
        errors.append("parse_json_response failed on 0.5 score")

    sample = {
        "sample_id": "100K/1",
        "size": "100K",
        "conversation_id": "1",
        "chat": [
            {
                "batch_number": 1,
                "time_anchor": "March-15-2024",
                "turns": [
                    [
                        {
                            "role": "user",
                            "content": "Hello",
                            "question_type": "main_question",
                            "time_anchor": "March-15-2024",
                        },
                        {"role": "assistant", "content": "Hi there"},
                    ]
                ],
            }
        ],
        "probing_questions": {
            "information_extraction": [
                {"question": "What was said?", "rubric": ["Hello"]}
            ]
        },
    }
    units = iter_ingest_units(sample)
    if len(units) != 1:
        errors.append(f"expected 1 ingest unit, got {len(units)}")
    elif "March 15, 2024" not in str(units[0].get("source_timestamp")):
        errors.append(f"unexpected source_timestamp: {units[0].get('source_timestamp')}")

    if errors:
        for err in errors:
            console.print(f"[red]FAIL[/red] {err}")
        return 1
    console.print("[green]selftest passed[/green]")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="beam",
        description="BEAM long-term memory benchmark harness for BrainAPI",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=None,
        help="Optional path to .env (default: benchmarks/.env)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_download = sub.add_parser("download", help="Download and normalize BEAM from HF")
    p_download.add_argument(
        "--size",
        choices=CHAT_SIZES,
        default=None,
        help="Download one size only (default: all 100K/500K/1M/10M)",
    )
    p_download.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing normalized conversations",
    )
    p_download.set_defaults(func=cmd_download)

    p_stats = sub.add_parser("dataset-stats", help="Print local dataset stats")
    p_stats.add_argument("--size", choices=CHAT_SIZES, default=None)
    p_stats.set_defaults(func=cmd_dataset_stats)

    p_smoke = sub.add_parser("smoke", help="Ingest a few turns + retrieve")
    p_smoke.add_argument("--size", choices=CHAT_SIZES, default="100K")
    p_smoke.add_argument("--sample", default="1")
    p_smoke.add_argument("--brain", default=None)
    p_smoke.add_argument("--limit-turns", type=int, default=2)
    p_smoke.add_argument("--timeout", type=float, default=900.0)
    p_smoke.set_defaults(func=cmd_smoke)

    p_ingest = sub.add_parser("ingest", help="Ingest BEAM turns into BrainAPI")
    p_ingest.add_argument("--size", choices=CHAT_SIZES, required=True)
    p_ingest.add_argument(
        "--sample",
        action="append",
        default=None,
        help="Conversation id or size/id (repeatable)",
    )
    p_ingest.add_argument("--run", default=None)
    p_ingest.add_argument("--brain", default=None)
    p_ingest.add_argument(
        "--concurrency",
        type=int,
        default=2,
        help=(
            "Parallel ingest units (default 2). For BEAM 1M prefer 2–3; "
            "≤4 recommended. Ensure CELERY_WORKER_CONCURRENCY ≥ this value."
        ),
    )
    p_ingest.add_argument("--limit-turns", type=int, default=None)
    p_ingest.add_argument("--timeout", type=float, default=900.0)
    p_ingest.add_argument("--dry-run", action="store_true")
    p_ingest.add_argument("--no-resume", action="store_true")
    p_ingest.set_defaults(func=cmd_ingest)

    p_eval = sub.add_parser("evaluate", help="Retrieve → answer → rubric-judge")
    p_eval.add_argument("--size", choices=CHAT_SIZES, required=True)
    p_eval.add_argument("--sample", action="append", default=None)
    p_eval.add_argument("--run", default=None)
    p_eval.add_argument("--brain", default=None)
    p_eval.add_argument("--concurrency", type=int, default=2)
    p_eval.add_argument(
        "--abilities",
        default=None,
        help=f"Comma-separated abilities from: {','.join(ABILITY_NAMES)}",
    )
    p_eval.add_argument("--limit", type=int, default=None)
    p_eval.add_argument("--no-resume", action="store_true")
    p_eval.add_argument(
        "--profile",
        choices=("product", "sota"),
        default=None,
        help="Override BENCH_PROFILE for this evaluate run",
    )
    p_eval.add_argument(
        "--historical-limit", type=int, default=DEFAULT_HISTORICAL_LIMIT
    )
    p_eval.add_argument("--max-passages", type=int, default=DEFAULT_MAX_PASSAGES)
    p_eval.add_argument("--max-facts", type=int, default=DEFAULT_MAX_FACTS)
    p_eval.add_argument("--no-fact-filter", action="store_true")
    p_eval.add_argument(
        "--use-ppr",
        action="store_true",
        default=True,
        help="Enable PPR re-ranking (default on; use --no-ppr to disable)",
    )
    p_eval.add_argument(
        "--no-ppr",
        action="store_true",
        help="Disable PPR re-ranking",
    )
    p_eval.add_argument("--sufficiency-retry", action="store_true")
    add_max_degraded_rate_argument(p_eval)
    p_eval.set_defaults(func=cmd_evaluate)

    p_report = sub.add_parser("report", help="Rebuild report from answers.jsonl")
    p_report.add_argument("--run", required=True)
    add_max_degraded_rate_argument(p_report)
    p_report.set_defaults(func=cmd_report)

    p_self = sub.add_parser("selftest", help="Local unit checks (no API)")
    p_self.set_defaults(func=cmd_selftest)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    settings = Settings.load(args.env_file)
    return int(args.func(args, settings))
