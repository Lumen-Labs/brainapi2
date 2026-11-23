from __future__ import annotations

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from beam.config import ABILITY_NAMES, BENCHMARKS_ROOT
from beam.evaluate import row_is_degraded
from beam.ingest import load_jsonl
from beam.metrics import aggregate_answers

console = Console()

REPORTS_PATH = BENCHMARKS_ROOT / "REPORTS.json"
_BENCHMARK_ID = "beam"
_BENCHMARK_NAME = "BEAM"
_REPORTS_DESCRIPTION = (
    "BrainAPI benchmark results. Top published scores across suites. "
    "Updated when a suite evaluate/report completes successfully."
)
_LEADERBOARD_ENTRY_KEYS = (
    "run_id",
    "track",
    "bench_profile",
    "scope",
    "size",
    "samples",
    "n_questions",
    "headline_score",
    "headline_score_pct",
    "per_ability",
    "answer_model",
    "judge_model",
    "brain",
    "git_sha",
    "report_path",
    "recorded_at",
)
_MIN_LEADERBOARD_N = 20


def _latest_ingest_rows(ingest_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    for row in ingest_rows:
        key = (
            str(row.get("sample_id") or ""),
            str(row.get("unit_id") or row.get("session_key") or ""),
        )
        latest[key] = row
    return list(latest.values())


def _latest_answer_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    latest: dict[tuple[str, str, int], dict[str, Any]] = {}
    for row in rows:
        key = (
            str(row.get("sample_id") or ""),
            str(row.get("ability") or ""),
            int(row.get("qa_index") or 0),
        )
        latest[key] = row
    return list(latest.values())


def build_report(run_dir: Path, *, include_degraded: bool = False) -> dict[str, Any]:
    answers_path = run_dir / "answers.jsonl"
    ingest_path = run_dir / "ingest.jsonl"
    manifest_path = run_dir / "manifest.json"

    all_answer_rows = load_jsonl(answers_path)
    scored_rows = [r for r in all_answer_rows if not r.get("error")]
    answers = _latest_answer_rows(scored_rows)
    errored = [r for r in all_answer_rows if r.get("error")]
    duplicates_dropped = len(scored_rows) - len(answers)
    degraded_answers = [r for r in answers if row_is_degraded(r)]
    scored_answers = (
        answers if include_degraded else [r for r in answers if not row_is_degraded(r)]
    )
    empty_predictions = sum(
        1 for r in scored_answers if not str(r.get("prediction") or "").strip()
    )
    ingest_rows = load_jsonl(ingest_path)
    ingest_latest = _latest_ingest_rows(ingest_rows)
    manifest = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    metrics = aggregate_answers(scored_answers)
    ingest_summary = {
        "n": len(ingest_latest),
        "completed": sum(1 for r in ingest_latest if r.get("status") == "completed"),
        "partial_failed": sum(
            1 for r in ingest_latest if r.get("status") == "partial_failed"
        ),
        "failed": sum(1 for r in ingest_latest if r.get("status") == "failed"),
        "dry_run": sum(1 for r in ingest_latest if r.get("status") == "dry_run"),
        "attempts": len(ingest_rows),
    }

    integrity = {
        "rows_in_file": len(all_answer_rows),
        "errored_rows": len(errored),
        "scored_rows": len(scored_answers),
        "duplicates_dropped": duplicates_dropped,
        "empty_predictions": empty_predictions,
        "degraded_rows": len(degraded_answers),
        "include_degraded": include_degraded,
    }

    report: dict[str, Any] = {
        "run_dir": str(run_dir),
        "manifest": manifest,
        "ingest": ingest_summary,
        "integrity": integrity,
        "metrics": metrics,
        "status": "ok",
        "warnings": [],
    }
    if ingest_summary["partial_failed"] or ingest_summary["failed"]:
        report["warnings"].append(
            "Some ingest units failed or partially failed; scores may be skewed."
        )
    if duplicates_dropped:
        report["warnings"].append(
            f"{duplicates_dropped} duplicate answer rows dropped."
        )
    if errored:
        report["warnings"].append(
            f"{len(errored)} rows errored and are excluded from every metric."
        )
    if degraded_answers and not include_degraded:
        report["warnings"].append(
            f"{len(degraded_answers)} rows had degraded retrieval and are "
            "excluded from accuracy scores."
        )

    failure_reasons: list[str] = []
    if not scored_answers:
        failure_reasons.append("no answer rows were scored")
    else:
        if empty_predictions == len(scored_answers):
            failure_reasons.append("every scored answer is empty")
        if metrics.get("headline_score") == 0.0:
            failure_reasons.append("headline score is 0.0 across scored abilities")
    if failure_reasons:
        report["status"] = "failed"
        report["warnings"].insert(
            0,
            "RUN FAILED WHOLESALE: "
            + "; ".join(failure_reasons)
            + ". The percentages below are not a measurement of quality.",
        )
    elif errored or duplicates_dropped or (degraded_answers and not include_degraded):
        report["status"] = "degraded"
    return report


def render_markdown(report: dict[str, Any]) -> str:
    metrics = report["metrics"]
    manifest = report.get("manifest") or {}
    integrity = report.get("integrity") or {}
    lines = ["# BEAM BrainAPI Benchmark Report", ""]
    if report.get("status") == "failed":
        lines.extend(
            [
                "> **RUN FAILED WHOLESALE — the numbers below are not a measurement.**",
                f"> {report['warnings'][0]}",
                "",
            ]
        )
    elif report.get("status") == "degraded":
        lines.extend(
            [
                "> **Run degraded** — errored or duplicate rows were dropped; "
                "see Warnings.",
                "",
            ]
        )
    lines.extend(
        [
            f"- Run dir: `{report['run_dir']}`",
            f"- Created: {manifest.get('created_at', 'n/a')}",
            f"- Size: {manifest.get('size', 'n/a')}",
            f"- Answer model: {manifest.get('answer_model', 'n/a')}",
            f"- Judge model: {manifest.get('judge_model', 'n/a')} "
            f"(provider: {manifest.get('judge_provider', 'n/a')}, "
            f"same family as answerer: "
            f"{manifest.get('judge_shares_answer_family', 'n/a')})",
            f"- Judge prompt variant: {manifest.get('judge_prompt_variant', 'n/a')}",
            f"- Git SHA: {manifest.get('git_sha', 'n/a')} "
            f"(dirty: {manifest.get('git_dirty', 'n/a')})",
            f"- Samples: {', '.join(manifest.get('samples', []) or ['n/a'])}",
            "",
            "## Run integrity",
            "",
            f"- Rows in answers.jsonl: {integrity.get('rows_in_file', 'n/a')}",
            f"- Scored unique questions: {integrity.get('scored_rows', 'n/a')}",
            f"- Errored rows (excluded): {integrity.get('errored_rows', 'n/a')}",
            f"- Degraded retrieval rows: {integrity.get('degraded_rows', 0)}",
            f"- Duplicate rows dropped: {integrity.get('duplicates_dropped', 'n/a')}",
            f"- Empty predictions: {integrity.get('empty_predictions', 'n/a')}",
            "",
            "## Headline",
            "",
            f"- Mean ability score: **{_pct(metrics.get('headline_score'))}** "
            f"(n_questions={metrics.get('n_questions', 0)}, "
            f"abilities={metrics.get('n_abilities_scored', 0)})",
            "",
            "## Per-ability",
            "",
            "| Ability | N | Mean |",
            "|---|---:|---:|",
        ]
    )
    for ability in ABILITY_NAMES:
        bucket = (metrics.get("per_ability") or {}).get(ability) or {}
        lines.append(
            f"| {ability} | {bucket.get('n', 0)} | {_pct(bucket.get('mean'))} |"
        )

    lat = metrics.get("retrieval_latency_ms") or {}
    lines.extend(
        [
            "",
            "## Latency & tokens",
            "",
            f"- Retrieval latency p50: {_num(lat.get('p50'))} ms",
            f"- Retrieval latency p95: {_num(lat.get('p95'))} ms",
            f"- Retrieval latency mean: {_num(lat.get('mean'))} ms",
            f"- Total LLM tokens (answer+judge): {metrics.get('total_llm_tokens', 0)}",
            "",
            "## Ingest",
            "",
            f"- Units: {report['ingest'].get('n', 0)}",
            f"- Completed: {report['ingest'].get('completed', 0)}",
            f"- Partial failed: {report['ingest'].get('partial_failed', 0)}",
            f"- Failed: {report['ingest'].get('failed', 0)}",
            "",
        ]
    )
    if report.get("warnings"):
        lines.append("## Warnings")
        lines.append("")
        for warning in report["warnings"]:
            lines.append(f"- {warning}")
        lines.append("")
    return "\n".join(lines)


def entry_from_report(report: dict[str, Any]) -> dict[str, Any] | None:
    if report.get("status") == "failed":
        return None
    metrics = report.get("metrics") or {}
    headline = metrics.get("headline_score")
    n_questions = int(metrics.get("n_questions") or 0)
    if headline is None or n_questions < _MIN_LEADERBOARD_N:
        return None

    manifest = report.get("manifest") or {}
    run_id = str(manifest.get("run_id") or "").strip()
    if not run_id:
        run_dir = str(report.get("run_dir") or "").rstrip("/")
        run_id = Path(run_dir).name if run_dir else ""
    if not run_id:
        return None

    samples = list(manifest.get("samples") or [])
    size = str(manifest.get("size") or "unknown")
    if len(samples) == 1:
        scope = str(samples[0])
    elif samples:
        scope = f"{size}:{len(samples)}"
    else:
        scope = size

    per_ability = {
        ability: ((metrics.get("per_ability") or {}).get(ability) or {}).get("mean")
        for ability in ABILITY_NAMES
    }
    acc = float(headline)
    now = datetime.now(timezone.utc).isoformat()
    bench_profile = manifest.get("bench_profile") or "product"
    track = "sota" if bench_profile == "sota" else "product"
    return {
        "run_id": run_id,
        "track": track,
        "bench_profile": bench_profile,
        "scope": scope,
        "size": size,
        "samples": samples,
        "n_questions": n_questions,
        "headline_score": acc,
        "headline_score_pct": round(acc * 100, 2),
        "per_ability": per_ability,
        "answer_model": manifest.get("answer_model"),
        "judge_model": manifest.get("judge_model"),
        "brain": manifest.get("brain_override"),
        "git_sha": manifest.get("git_sha"),
        "report_path": f"runs/{run_id}/report.json",
        "recorded_at": now,
    }


def _empty_reports_ledger() -> dict[str, Any]:
    return {
        "schema_version": 2,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "description": _REPORTS_DESCRIPTION,
        "benchmarks": {
            _BENCHMARK_ID: {
                "name": _BENCHMARK_NAME,
                "leaderboard": [],
            }
        },
    }


def _public_leaderboard_entry(row: dict[str, Any]) -> dict[str, Any]:
    return {key: row[key] for key in _LEADERBOARD_ENTRY_KEYS if key in row}


def _sorted_leaderboard(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    leaderboard = [
        _public_leaderboard_entry(row)
        for row in rows
        if isinstance(row, dict) and row.get("run_id")
    ]
    leaderboard.sort(
        key=lambda row: float(row.get("headline_score") or 0.0),
        reverse=True,
    )
    return leaderboard


def _normalize_reports_ledger(data: dict[str, Any]) -> dict[str, Any]:
    benchmarks: dict[str, Any] = {}
    raw_benchmarks = data.get("benchmarks")
    if isinstance(raw_benchmarks, dict):
        for suite_id, suite in raw_benchmarks.items():
            if not isinstance(suite, dict):
                continue
            if str(suite_id) == _BENCHMARK_ID:
                leaderboard = _sorted_leaderboard(list(suite.get("leaderboard") or []))
            else:
                leaderboard = list(suite.get("leaderboard") or [])
            benchmarks[str(suite_id)] = {
                "name": suite.get("name") or str(suite_id),
                "leaderboard": leaderboard,
            }

    if _BENCHMARK_ID not in benchmarks:
        benchmarks[_BENCHMARK_ID] = {
            "name": _BENCHMARK_NAME,
            "leaderboard": [],
        }

    return {
        "schema_version": 2,
        "updated_at": data.get("updated_at")
        or datetime.now(timezone.utc).isoformat(),
        "description": data.get("description") or _REPORTS_DESCRIPTION,
        "benchmarks": benchmarks,
    }


def update_reports_json(report: dict[str, Any]) -> None:
    entry = entry_from_report(report)
    if entry is None:
        return

    if REPORTS_PATH.exists():
        data = json.loads(REPORTS_PATH.read_text(encoding="utf-8"))
    else:
        data = _empty_reports_ledger()

    data = _normalize_reports_ledger(data)
    suite = data["benchmarks"].setdefault(
        _BENCHMARK_ID,
        {"name": _BENCHMARK_NAME, "leaderboard": []},
    )
    run_id = entry["run_id"]
    leaderboard = list(suite.get("leaderboard") or [])
    lb_idx = next(
        (i for i, row in enumerate(leaderboard) if row.get("run_id") == run_id),
        None,
    )
    if lb_idx is not None:
        leaderboard[lb_idx] = entry
    else:
        leaderboard.append(entry)

    suite["name"] = suite.get("name") or _BENCHMARK_NAME
    suite["leaderboard"] = leaderboard
    data["benchmarks"][_BENCHMARK_ID] = suite
    data["updated_at"] = datetime.now(timezone.utc).isoformat()
    data = _normalize_reports_ledger(data)

    REPORTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix="REPORTS.",
        suffix=".json.tmp",
        dir=str(REPORTS_PATH.parent),
    )
    tmp_path = Path(tmp_name)
    try:
        with open(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
            fh.write("\n")
        tmp_path.replace(REPORTS_PATH)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def write_report(run_dir: Path, *, include_degraded: bool = False) -> dict[str, Any]:
    report = build_report(run_dir, include_degraded=include_degraded)
    md = render_markdown(report)
    (run_dir / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    (run_dir / "report.md").write_text(md, encoding="utf-8")
    if report.get("status") == "ok":
        update_reports_json(report)
    return report


def print_report_table(report: dict[str, Any]) -> None:
    metrics = report["metrics"]
    table = Table(title="BEAM per-ability results")
    table.add_column("Ability")
    table.add_column("N", justify="right")
    table.add_column("Mean", justify="right")
    for ability in ABILITY_NAMES:
        bucket = (metrics.get("per_ability") or {}).get(ability) or {}
        table.add_row(
            ability,
            str(bucket.get("n", 0)),
            _pct(bucket.get("mean")),
        )
    if report.get("status") == "failed":
        console.print(
            f"[bold red]{(report.get('warnings') or ['RUN FAILED'])[0]}[/bold red]"
        )
    console.print(table)
    console.print(
        f"Headline mean ability score: "
        f"[bold]{_pct(metrics.get('headline_score'))}[/bold] "
        f"(n={metrics.get('n_questions', 0)})"
    )
    for warning in report.get("warnings") or []:
        console.print(f"[yellow]Warning: {warning}[/yellow]")


def _pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.1f}%"


def _num(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}"
