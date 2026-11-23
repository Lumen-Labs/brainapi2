from __future__ import annotations

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from degraded import DEFAULT_MAX_DEGRADED_RATE, score_degraded_rows
from longmemeval.config import BENCHMARKS_ROOT
from longmemeval.evaluate import row_is_degraded
from longmemeval.ingest import load_jsonl
from longmemeval.metrics import aggregate_answers

console = Console()

REPORTS_PATH = BENCHMARKS_ROOT / "REPORTS.json"
_BENCHMARK_ID = "longmemeval"
_BENCHMARK_NAME = "LongMemEval"
_REPORTS_DESCRIPTION = (
    "BrainAPI benchmark results. Top published scores across suites. "
    "Updated when a suite evaluate/report completes successfully."
)
_LEADERBOARD_ENTRY_KEYS = (
    "run_id",
    "track",
    "scope",
    "variant",
    "n_questions",
    "headline_judge_accuracy",
    "headline_judge_accuracy_pct",
    "answer_model",
    "judge_model",
    "bench_profile",
    "git_sha",
    "report_path",
    "recorded_at",
)
_MIN_LEADERBOARD_N = 50


def _latest_ingest_rows(ingest_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    for row in ingest_rows:
        key = (
            str(row.get("question_id") or ""),
            str(row.get("unit_id") or row.get("session_id") or ""),
        )
        latest[key] = row
    return list(latest.values())


def _latest_answer_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = str(row.get("question_id") or "")
        if key:
            latest[key] = row
    return list(latest.values())


def build_report(
    run_dir: Path,
    *,
    include_degraded: bool = False,
    max_degraded_rate: float = DEFAULT_MAX_DEGRADED_RATE,
) -> dict[str, Any]:
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
    truncated_rows = sum(1 for r in scored_answers if r.get("context_truncated"))
    ingest_rows = load_jsonl(ingest_path)
    ingest_latest = _latest_ingest_rows(ingest_rows)
    manifest = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    metrics = aggregate_answers(scored_answers)
    degraded_stats = score_degraded_rows(
        answers, max_degraded_rate=max_degraded_rate
    )
    metrics["headline_judge_accuracy_strict"] = degraded_stats["strict_accuracy"]
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
    wait_vals = [
        float(r["wait_latency_ms"])
        for r in ingest_latest
        if r.get("wait_latency_ms") is not None
    ]
    token_vals = [
        int(r["total_llm_tokens"])
        for r in ingest_latest
        if r.get("total_llm_tokens") is not None
    ]
    if wait_vals:
        ordered = sorted(wait_vals)
        mid = len(ordered) // 2
        ingest_summary["wait_latency_p50_ms"] = (
            ordered[mid]
            if len(ordered) % 2
            else (ordered[mid - 1] + ordered[mid]) / 2
        )
        ingest_summary["wait_latency_mean_ms"] = sum(ordered) / len(ordered)
    if token_vals:
        ingest_summary["mean_ingest_llm_tokens"] = sum(token_vals) / len(token_vals)
        ingest_summary["total_ingest_llm_tokens"] = sum(token_vals)
        stage_keys = (
            "scout_total_tokens",
            "architect_total_tokens",
            "janitor_total_tokens",
            "observations_total_tokens",
            "consolidation_total_tokens",
        )
        stage_sums = {k: 0 for k in stage_keys}
        stage_n = 0
        for row in ingest_latest:
            cost = row.get("cost") or {}
            stages = cost.get("stages") if isinstance(cost, dict) else None
            if not isinstance(stages, dict):
                continue
            stage_n += 1
            for key in stage_keys:
                stage_name = key.replace("_total_tokens", "")
                stage_sums[key] += int(
                    (stages.get(stage_name) or {}).get("total_tokens") or 0
                )
        if stage_n:
            ingest_summary["mean_tokens_by_stage"] = {
                k.replace("_total_tokens", ""): stage_sums[k] / stage_n
                for k in stage_keys
            }

    integrity = {
        "rows_in_file": len(all_answer_rows),
        "errored_rows": len(errored),
        "scored_rows": len(scored_answers),
        "duplicates_dropped": duplicates_dropped,
        "empty_predictions": empty_predictions,
        "rows_with_truncated_context": truncated_rows,
        "degraded_rows": len(degraded_answers),
        "degraded_rate": degraded_stats["degraded_rate"],
        "max_degraded_rate": max_degraded_rate,
        "strict_accuracy": degraded_stats["strict_accuracy"],
        "exceeded_max_degraded_rate": degraded_stats["exceeded_max_degraded_rate"],
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
    if degraded_stats["exceeded_max_degraded_rate"]:
        rate = degraded_stats["degraded_rate"]
        report["warnings"].append(
            f"Degraded retrieval rate {rate:.1%} exceeds max "
            f"{max_degraded_rate:.1%}."
        )
    if truncated_rows:
        report["warnings"].append(
            f"{truncated_rows} rows had a retrieved-context channel truncated "
            "before logging."
        )

    failure_reasons: list[str] = []
    if not scored_answers:
        failure_reasons.append("no answer rows were scored")
    else:
        if empty_predictions == len(scored_answers):
            failure_reasons.append("every scored answer is empty")
        if metrics.get("headline_judge_accuracy") == 0.0:
            failure_reasons.append("every scored answer was judged incorrect")
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


def _pct(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{100.0 * float(value):.1f}%"


def _num(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.2f}"


def _ci(ci: Any) -> str:
    if not isinstance(ci, dict) or ci.get("low") is None or ci.get("high") is None:
        return ""
    return f"[{100.0 * float(ci['low']):.1f}%, {100.0 * float(ci['high']):.1f}%]"


def render_markdown(report: dict[str, Any]) -> str:
    metrics = report["metrics"]
    manifest = report.get("manifest") or {}
    integrity = report.get("integrity") or {}
    lines = [
        "# LongMemEval BrainAPI Benchmark Report",
        "",
    ]
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
            f"- Variant: {manifest.get('variant', 'n/a')}",
            f"- Answer model: {manifest.get('answer_model', 'n/a')}",
            f"- Judge model: {manifest.get('judge_model', 'n/a')} "
            f"(provider: {manifest.get('judge_provider', 'n/a')}, "
            f"same family as answerer: "
            f"{manifest.get('judge_shares_answer_family', 'n/a')})",
            f"- Git SHA: {manifest.get('git_sha', 'n/a')} "
            f"(dirty: {manifest.get('git_dirty', 'n/a')})",
            f"- Answer prompt sha256: {manifest.get('answer_prompt_sha256', 'n/a')}",
            f"- Judge prompt sha256: {manifest.get('judge_prompt_sha256', 'n/a')}",
            f"- Dataset sha256: {manifest.get('dataset_sha256', 'n/a')}",
            "",
            "## Run integrity",
            "",
            f"- Rows in answers.jsonl: {integrity.get('rows_in_file', 'n/a')}",
            f"- Scored unique questions: {integrity.get('scored_rows', 'n/a')}",
            f"- Errored rows (excluded): {integrity.get('errored_rows', 'n/a')}",
            f"- Degraded retrieval rows: {integrity.get('degraded_rows', 0)} "
            f"(rate {_pct(integrity.get('degraded_rate'))}, "
            f"max {_pct(integrity.get('max_degraded_rate'))})",
            f"- Duplicate rows dropped: {integrity.get('duplicates_dropped', 'n/a')}",
            f"- Empty predictions: {integrity.get('empty_predictions', 'n/a')}",
            "",
            "## Headline",
            "",
            f"- Judge accuracy (all questions, incl. abstention): "
            f"**{_pct(metrics.get('headline_judge_accuracy'))}** "
            f"{_ci(metrics.get('headline_judge_accuracy_ci95'))} "
            f"(n={metrics.get('n_questions', 0)})",
            f"- Strict judge accuracy (degraded counted wrong): "
            f"**{_pct(metrics.get('headline_judge_accuracy_strict'))}**",
            f"- Session recall (full, excl. abstention): "
            f"**{_pct(metrics.get('session_recall_full'))}** "
            f"(n={metrics.get('n_with_evidence', 0)})",
            f"- Session recall (partial+full): "
            f"**{_pct(metrics.get('session_recall_partial'))}**",
            f"- Abstention accuracy: "
            f"**{_pct((metrics.get('abstention') or {}).get('accuracy'))}** "
            f"(n={(metrics.get('abstention') or {}).get('n', 0)})",
            "",
            "## Per question type",
            "",
            "| Type | N | Judge Acc | 95% CI | Session recall full |",
            "|---|---:|---:|---|---:|",
        ]
    )
    for qtype, bucket in (metrics.get("by_type") or {}).items():
        lines.append(
            f"| {qtype} | {bucket.get('n', 0)} | "
            f"{_pct(bucket.get('judge_accuracy'))} | "
            f"{_ci(bucket.get('judge_accuracy_ci95'))} | "
            f"{_pct(bucket.get('session_recall_full'))} |"
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
            *(
                [
                    f"- Ingest wait latency p50: "
                    f"{_num(report['ingest'].get('wait_latency_p50_ms'))} ms"
                ]
                if report["ingest"].get("wait_latency_p50_ms") is not None
                else []
            ),
            *(
                [
                    f"- Mean ingest LLM tokens/unit: "
                    f"{_num(report['ingest'].get('mean_ingest_llm_tokens'))}"
                ]
                if report["ingest"].get("mean_ingest_llm_tokens") is not None
                else []
            ),
            *(
                [
                    f"- Total ingest LLM tokens: "
                    f"{report['ingest'].get('total_ingest_llm_tokens')}"
                ]
                if report["ingest"].get("total_ingest_llm_tokens") is not None
                else []
            ),
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
    headline = metrics.get("headline_judge_accuracy")
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

    variant = str(manifest.get("variant") or "s")
    limit = manifest.get("limit")
    if limit:
        scope = f"{variant}:limit{limit}"
    else:
        scope = variant

    bench_profile = manifest.get("bench_profile") or "product"
    track = "sota" if bench_profile == "sota" else "product"
    acc = float(headline)
    now = datetime.now(timezone.utc).isoformat()

    return {
        "run_id": run_id,
        "track": track,
        "scope": scope,
        "variant": variant,
        "n_questions": n_questions,
        "headline_judge_accuracy": acc,
        "headline_judge_accuracy_pct": round(acc * 100, 2),
        "answer_model": manifest.get("answer_model"),
        "judge_model": manifest.get("judge_model"),
        "bench_profile": bench_profile,
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
        key=lambda row: float(row.get("headline_judge_accuracy") or 0.0),
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
            benchmarks[str(suite_id)] = {
                "name": suite.get("name") or str(suite_id),
                "leaderboard": _sorted_leaderboard(list(suite.get("leaderboard") or [])),
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
    suite["name"] = _BENCHMARK_NAME
    suite["leaderboard"] = _sorted_leaderboard(leaderboard)
    data["benchmarks"][_BENCHMARK_ID] = suite
    data["updated_at"] = datetime.now(timezone.utc).isoformat()
    data["description"] = _REPORTS_DESCRIPTION

    REPORTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=str(REPORTS_PATH.parent),
        delete=False,
        suffix=".tmp",
    ) as tmp:
        json.dump(data, tmp, indent=2)
        tmp.write("\n")
        tmp_path = Path(tmp.name)
    tmp_path.replace(REPORTS_PATH)


def write_report(
    run_dir: Path,
    *,
    include_degraded: bool = False,
    max_degraded_rate: float = DEFAULT_MAX_DEGRADED_RATE,
) -> dict[str, Any]:
    report = build_report(
        run_dir,
        include_degraded=include_degraded,
        max_degraded_rate=max_degraded_rate,
    )
    (run_dir / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    (run_dir / "report.md").write_text(render_markdown(report), encoding="utf-8")
    if report.get("status") == "ok":
        update_reports_json(report)
    return report


def print_report_table(report: dict[str, Any]) -> None:
    metrics = report.get("metrics") or {}
    table = Table(title="LongMemEval results")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row(
        "Headline judge accuracy",
        f"{_pct(metrics.get('headline_judge_accuracy'))} "
        f"(n={metrics.get('n_questions', 0)})",
    )
    table.add_row(
        "Strict judge accuracy",
        _pct(metrics.get("headline_judge_accuracy_strict")),
    )
    integrity = report.get("integrity") or {}
    table.add_row(
        "Degraded retrieval",
        f"{integrity.get('degraded_rows', 0)} "
        f"({_pct(integrity.get('degraded_rate'))})",
    )
    table.add_row(
        "Session recall (full)",
        f"{_pct(metrics.get('session_recall_full'))} "
        f"(n={metrics.get('n_with_evidence', 0)})",
    )
    table.add_row(
        "Abstention accuracy",
        f"{_pct((metrics.get('abstention') or {}).get('accuracy'))} "
        f"(n={(metrics.get('abstention') or {}).get('n', 0)})",
    )
    table.add_row("Status", str(report.get("status")))
    console.print(table)
    by_type = metrics.get("by_type") or {}
    if by_type:
        type_table = Table(title="By question type")
        type_table.add_column("Type")
        type_table.add_column("N", justify="right")
        type_table.add_column("Acc")
        for qtype, bucket in by_type.items():
            type_table.add_row(
                qtype,
                str(bucket.get("n", 0)),
                _pct(bucket.get("judge_accuracy")),
            )
        console.print(type_table)
