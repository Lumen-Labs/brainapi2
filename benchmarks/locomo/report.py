from __future__ import annotations

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from locomo.config import BENCHMARKS_ROOT, CATEGORY_NAMES
from locomo.evaluate import row_is_degraded
from locomo.ingest import load_jsonl
from locomo.metrics import aggregate_answers

console = Console()

REPORTS_PATH = BENCHMARKS_ROOT / "REPORTS.json"
_BENCHMARK_ID = "locomo"
_BENCHMARK_NAME = "LoCoMo"
_REPORTS_DESCRIPTION = (
    "BrainAPI benchmark results. Top published scores across suites. "
    "Updated when a suite evaluate/report completes successfully."
)
_LEADERBOARD_ENTRY_KEYS = (
    "run_id",
    "track",
    "scope",
    "samples",
    "n_non_adversarial",
    "headline_judge_accuracy",
    "headline_judge_accuracy_pct",
    "answerable_rate",
    "answer_model",
    "judge_model",
    "bench_profile",
    "brain",
    "git_sha",
    "report_path",
    "recorded_at",
)
_MIN_LEADERBOARD_N = 50


def _latest_ingest_rows(ingest_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep the last status per sample/unit (resume appends retries)."""
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    for row in ingest_rows:
        key = (
            str(row.get("sample_id") or ""),
            str(row.get("unit_id") or row.get("session_key") or ""),
        )
        latest[key] = row
    return list(latest.values())


def _latest_answer_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    latest: dict[tuple[str, int], dict[str, Any]] = {}
    for row in rows:
        key = (str(row.get("sample_id") or ""), int(row.get("qa_index") or 0))
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
    truncated_rows = sum(1 for r in scored_answers if r.get("context_truncated"))
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
                stage_sums[key] += int((stages.get(stage_name) or {}).get("total_tokens") or 0)
        if stage_n:
            ingest_summary["mean_tokens_by_stage"] = {
                k.replace("_total_tokens", ""): stage_sums[k] / stage_n for k in stage_keys
            }

    integrity = {
        "rows_in_file": len(all_answer_rows),
        "errored_rows": len(errored),
        "scored_rows": len(scored_answers),
        "duplicates_dropped": duplicates_dropped,
        "empty_predictions": empty_predictions,
        "rows_with_truncated_context": truncated_rows,
        "degraded_rows": len(degraded_answers),
        "include_degraded": include_degraded,
    }

    report = {
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
            f"{duplicates_dropped} duplicate answer rows dropped "
            f"({len(scored_rows)} scored rows over {len(answers)} unique questions)."
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
    if truncated_rows:
        report["warnings"].append(
            f"{truncated_rows} rows had a retrieved-context channel truncated "
            "before logging; see context_truncated in answers.jsonl."
        )
    graph_rows = [r for r in answers if r.get("graph_context") or r.get("triples")]
    if graph_rows and not any(r.get("retrieved_session_ids_graph") for r in graph_rows):
        report["warnings"].append(
            f"The graph channel was logged for {len(graph_rows)} rows but carries no "
            "session ids, so its evidence recall reads 0% by construction: "
            "/retrieve/context returns graph facts without session provenance."
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


def render_markdown(report: dict[str, Any]) -> str:
    metrics = report["metrics"]
    manifest = report.get("manifest") or {}
    integrity = report.get("integrity") or {}
    channels = metrics.get("evidence_session_recall_by_channel") or {}
    lines = [
        "# LoCoMo BrainAPI Benchmark Report",
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
            f"- Granularity: {manifest.get('granularity', 'n/a')}",
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
            f"- Rows with truncated context: "
            f"{integrity.get('rows_with_truncated_context', 'n/a')}",
            "",
            "## Headline",
            "",
        ]
    )
    lines.extend([
        f"- Judge accuracy (excl. adversarial / cat 5): "
        f"**{_pct(metrics.get('headline_judge_accuracy'))}** "
        f"{_ci(metrics.get('headline_judge_accuracy_ci95'))} "
        f"(n={metrics.get('n_non_adversarial', 0)})",
        f"- Judge accuracy (all categories): "
        f"**{_pct(metrics.get('overall_judge_accuracy'))}** "
        f"{_ci(metrics.get('overall_judge_accuracy_ci95'))} "
        f"(n={metrics.get('n_total', 0)})",
        f"- Mean F1 (excl. adversarial): "
        f"**{_num(metrics.get('mean_f1_non_adversarial'))}**",
        f"- Mean BLEU-1 (excl. adversarial): "
        f"**{_num(metrics.get('mean_bleu1_non_adversarial'))}**",
        f"- Answerable rate (gold tokens in context): "
        f"**{_pct(metrics.get('answerable_rate'))}**",
        f"- Evidence-session recall (full): "
        f"**{_pct(metrics.get('evidence_session_recall_full'))}**",
        f"- Evidence-session recall (partial+full): "
        f"**{_pct(metrics.get('evidence_session_recall_partial'))}**",
        f"- Evidence-session recall, graph channel: "
        f"**{_pct((channels.get('graph') or {}).get('full'))}** "
        f"(n={(channels.get('graph') or {}).get('n_with_evidence', 0)})",
        f"- Evidence-session recall, passage channel: "
        f"**{_pct((channels.get('passages') or {}).get('full'))}** "
        f"(n={(channels.get('passages') or {}).get('n_with_evidence', 0)})",
        f"- Answerer gap (answerable − judge): "
        f"**{_pct(metrics.get('answerer_gap'))}**",
        f"- Abstention accuracy (cat 5): "
        f"**{_pct((metrics.get('abstention') or {}).get('accuracy'))}** "
        f"(n={(metrics.get('abstention') or {}).get('n', 0)})",
        "",
        "Retrieval-side metrics are deterministic for a fixed brain and config; "
        "they are reported exactly and carry no significance test. "
        "Judge accuracy is not — compare runs with `python -m locomo compare`.",
        "",
        "## Per-category",
        "",
        "| Category | Name | N | Judge Acc | 95% CI | Answerable | EvRecall | EvRecall graph | EvRecall passages | Mean F1 | Mean BLEU-1 |",
        "|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|",
    ])
    for cat, bucket in (metrics.get("by_category") or {}).items():
        name = CATEGORY_NAMES.get(int(cat), "unknown")
        lines.append(
            f"| {cat} | {name} | {bucket.get('n', 0)} | "
            f"{_pct(bucket.get('judge_accuracy'))} | "
            f"{_ci(bucket.get('judge_accuracy_ci95'))} | "
            f"{_pct(bucket.get('answerable_rate'))} | "
            f"{_pct(bucket.get('evidence_session_recall_full'))} | "
            f"{_pct(bucket.get('evidence_session_recall_full_graph'))} | "
            f"{_pct(bucket.get('evidence_session_recall_full_passages'))} | "
            f"{_num(bucket.get('mean_f1'))} | "
            f"{_num(bucket.get('mean_bleu1'))} |"
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
                [f"- Attempts (incl. retries): {report['ingest'].get('attempts')}"]
                if report["ingest"].get("attempts")
                and report["ingest"].get("attempts") != report["ingest"].get("n")
                else []
            ),
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
            *(
                [
                    "- Mean tokens by stage: "
                    + ", ".join(
                        f"{k}={_num(v)}"
                        for k, v in (
                            report["ingest"].get("mean_tokens_by_stage") or {}
                        ).items()
                    )
                ]
                if report["ingest"].get("mean_tokens_by_stage")
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
    n_non_adv = int(metrics.get("n_non_adversarial") or 0)
    if headline is None or n_non_adv < _MIN_LEADERBOARD_N:
        return None

    manifest = report.get("manifest") or {}
    run_id = str(manifest.get("run_id") or "").strip()
    if not run_id:
        run_dir = str(report.get("run_dir") or "").rstrip("/")
        run_id = Path(run_dir).name if run_dir else ""
    if not run_id:
        return None

    samples = list(manifest.get("samples") or [])
    if len(samples) == 1:
        scope = str(samples[0])
    elif len(samples) >= 10:
        scope = "locomo10"
    elif samples:
        scope = ",".join(str(s) for s in samples)
    else:
        scope = "unknown"

    bench_profile = manifest.get("bench_profile") or "product"
    track = "sota" if bench_profile == "sota" else "product"
    brain = manifest.get("brain_override")
    acc = float(headline)
    now = datetime.now(timezone.utc).isoformat()

    return {
        "run_id": run_id,
        "track": track,
        "scope": scope,
        "samples": samples,
        "n_non_adversarial": n_non_adv,
        "headline_judge_accuracy": acc,
        "headline_judge_accuracy_pct": round(acc * 100, 2),
        "answerable_rate": metrics.get("answerable_rate"),
        "answer_model": manifest.get("answer_model"),
        "judge_model": manifest.get("judge_model"),
        "bench_profile": bench_profile,
        "brain": brain,
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

    # One-shot migrate schema v1 flat leaderboard into benchmarks.locomo.
    legacy = data.get("leaderboard")
    if isinstance(legacy, list) and legacy:
        suite = dict(benchmarks.get(_BENCHMARK_ID) or {})
        by_id: dict[str, dict[str, Any]] = {
            str(row["run_id"]): row
            for row in (suite.get("leaderboard") or [])
            if isinstance(row, dict) and row.get("run_id")
        }
        for row in legacy:
            if not isinstance(row, dict) or not row.get("run_id"):
                continue
            rid = str(row["run_id"])
            if rid not in by_id:
                by_id[rid] = _public_leaderboard_entry(row)
        suite["name"] = suite.get("name") or _BENCHMARK_NAME
        suite["leaderboard"] = _sorted_leaderboard(list(by_id.values()))
        benchmarks[_BENCHMARK_ID] = suite

    if _BENCHMARK_ID not in benchmarks:
        benchmarks[_BENCHMARK_ID] = {
            "name": _BENCHMARK_NAME,
            "leaderboard": [],
        }

    return {
        "schema_version": 2,
        "updated_at": data.get("updated_at")
        or datetime.now(timezone.utc).isoformat(),
        "description": _REPORTS_DESCRIPTION,
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
    data["description"] = _REPORTS_DESCRIPTION
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
    table = Table(title="LoCoMo per-category results")
    table.add_column("Cat")
    table.add_column("Name")
    table.add_column("N", justify="right")
    table.add_column("Judge Acc", justify="right")
    table.add_column("95% CI")
    table.add_column("Answerable", justify="right")
    table.add_column("EvRecall", justify="right")
    table.add_column("EvR graph", justify="right")
    table.add_column("EvR passages", justify="right")
    table.add_column("F1", justify="right")
    table.add_column("BLEU-1", justify="right")
    for cat, bucket in (metrics.get("by_category") or {}).items():
        table.add_row(
            str(cat),
            CATEGORY_NAMES.get(int(cat), "unknown"),
            str(bucket.get("n", 0)),
            _pct(bucket.get("judge_accuracy")),
            _ci(bucket.get("judge_accuracy_ci95")),
            _pct(bucket.get("answerable_rate")),
            _pct(bucket.get("evidence_session_recall_full")),
            _pct(bucket.get("evidence_session_recall_full_graph")),
            _pct(bucket.get("evidence_session_recall_full_passages")),
            _num(bucket.get("mean_f1")),
            _num(bucket.get("mean_bleu1")),
        )
    if report.get("status") == "failed":
        console.print(
            f"[bold red]{(report.get('warnings') or ['RUN FAILED'])[0]}[/bold red]"
        )
    console.print(table)
    console.print(
        f"Headline judge accuracy (excl. cat 5): "
        f"[bold]{_pct(metrics.get('headline_judge_accuracy'))}[/bold] "
        f"{_ci(metrics.get('headline_judge_accuracy_ci95'))}"
    )
    channels = metrics.get("evidence_session_recall_by_channel") or {}
    console.print(
        f"Answerable rate: {_pct(metrics.get('answerable_rate'))} | "
        f"EvRecall full: {_pct(metrics.get('evidence_session_recall_full'))} "
        f"(graph {_pct((channels.get('graph') or {}).get('full'))}, "
        f"passages {_pct((channels.get('passages') or {}).get('full'))}) | "
        f"Answerer gap: {_pct(metrics.get('answerer_gap'))}"
    )
    for warning in report.get("warnings") or []:
        console.print(f"[yellow]Warning: {warning}[/yellow]")


def print_comparison(
    comparison: dict[str, Any],
    baseline_label: str,
    candidate_label: str,
    retrieval: dict[str, Any] | None = None,
) -> None:
    overall = comparison["overall"]
    console.print(
        f"[bold]{baseline_label}[/bold] -> [bold]{candidate_label}[/bold] "
        f"({comparison['baseline_runs']} vs {comparison['candidate_runs']} run(s), "
        f"n={overall['n_paired']} paired questions)"
    )
    table = Table(title="Paired judge-accuracy comparison (exact McNemar)")
    table.add_column("Scope")
    table.add_column("N", justify="right")
    table.add_column("Flipped right", justify="right")
    table.add_column("Flipped wrong", justify="right")
    table.add_column("Exact p", justify="right")
    table.add_column("Significant")
    rows = [("overall", overall)]
    for cat, bucket in (comparison.get("by_category") or {}).items():
        rows.append((f"{cat} {CATEGORY_NAMES.get(int(cat), 'unknown')}", bucket))
    for label, bucket in rows:
        table.add_row(
            label,
            str(bucket["n_paired"]),
            str(bucket["flipped_right"]),
            str(bucket["flipped_wrong"]),
            f"{bucket['mcnemar_exact_p']:.4g}",
            "yes" if bucket["significant_at_05"] else "no",
        )
    console.print(table)
    console.print(
        f"Accuracy: {_pct(comparison['baseline_accuracy'])} -> "
        f"{_pct(comparison['candidate_accuracy'])}"
    )
    if retrieval:
        console.print(
            "[dim]Passage retrieval metrics are exact for a fixed brain/config. "
            "Graph EvR is a measurement only after graph-session set agreement "
            "clears the ≥95% gate below.[/dim]"
        )
        ret_table = Table(title="Retrieval metrics (exact, no significance test)")
        ret_table.add_column("Arm")
        ret_table.add_column("Run")
        ret_table.add_column("EvRecall full", justify="right")
        ret_table.add_column("EvR graph", justify="right")
        ret_table.add_column("EvR passages", justify="right")
        ret_table.add_column("Answerable", justify="right")
        for arm_label, arm in (
            ("baseline", retrieval.get("baseline") or {}),
            ("candidate", retrieval.get("candidate") or {}),
        ):
            for name, run in zip(
                arm.get("labels") or [], arm.get("per_run") or []
            ):
                channels = run.get("evidence_session_recall_by_channel") or {}
                ret_table.add_row(
                    arm_label,
                    name,
                    _pct(run.get("evidence_session_recall_full")),
                    _pct((channels.get("graph") or {}).get("full")),
                    _pct((channels.get("passages") or {}).get("full")),
                    _pct(run.get("answerable_rate")),
                )
        console.print(ret_table)
        for arm_label, arm in (
            ("baseline", retrieval.get("baseline") or {}),
            ("candidate", retrieval.get("candidate") or {}),
        ):
            if len(arm.get("per_run") or []) > 1:
                verdict = (
                    "identical across runs"
                    if arm.get("identical_across_runs")
                    else "NOT identical across runs"
                )
                console.print(f"[dim]{arm_label} retrieval metrics: {verdict}[/dim]")

    stability = comparison.get("graph_session_stability")
    if stability:
        rate = stability.get("agreement_rate")
        gate = stability.get("gate")
        n = stability.get("n_paired")
        identical = stability.get("identical_session_sets")
        passes = stability.get("passes_gate")
        console.print(
            f"Graph-session set agreement: "
            f"{_pct(rate)} ({identical}/{n} questions identical); "
            f"gate {_pct(gate)} — "
            f"{'PASS' if passes else 'FAIL (do not A/B graph EvR)'}"
        )
        cov_rate = stability.get("coverage_agreement_rate")
        if cov_rate is not None:
            console.print(
                f"Graph coverage-label agreement: {_pct(cov_rate)} "
                f"({stability.get('identical_graph_coverage')}/"
                f"{stability.get('n_coverage_comparable')})"
            )
        console.print(f"[dim]{stability.get('note')}[/dim]")
        disagreeing = stability.get("disagreeing_questions") or []
        if disagreeing:
            preview = ", ".join(disagreeing[:12])
            more = (
                f" (+{len(disagreeing) - 12} more)" if len(disagreeing) > 12 else ""
            )
            console.print(f"Disagreeing questions: {preview}{more}")


def _pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.1f}%"


def _ci(value: dict[str, float] | None) -> str:
    if not value or value.get("low") is None or value.get("high") is None:
        return "n/a"
    return f"[{_pct(value['low'])}, {_pct(value['high'])}]"


def _num(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}"
