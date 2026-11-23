from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from evals.config import Settings


def _pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{100.0 * value:.1f}%"


def _ms(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.0f} ms"


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    line = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([line, sep, *body])


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# Use-case eval report `{report.get('run_id')}`",
        "",
        f"- Scale: **{report.get('scale')}**",
        f"- Split: `{report.get('split')}`",
        f"- Baseline tier: `{report.get('baseline_tier') or ('smoke_hash' if report.get('smoke') else 'strong')}`",
        f"- Baseline profile: `{report.get('baseline_profile') or (report.get('env') or {}).get('baseline_profile') or 'sota'}`",
        f"- Dense: `{report.get('embedding_model') or (report.get('env') or {}).get('embedding_model') or ''}`",
        f"- Reranker: `{report.get('reranker_model') or (report.get('env') or {}).get('reranker_model') or ''}`",
        f"- Answerer: `{report.get('answerer')}`",
        f"- LLM: `{report.get('llm_model') or (report.get('env') or {}).get('llm_model') or ''}` (shared answerer, judge, and any BrainAPI LLM calls)",
        (
            "- Extractive scoring is a **retrieval ceiling** (gold facts in the packed window), "
            "not LLM answer quality. Use `--answerer llm` for the shared generative answerer."
            if report.get("answerer") == "extractive"
            else "- LLM answering uses the shared prompt budget for every system."
        ),
        f"- Prompt budget: `{json.dumps(report.get('prompt_budget'))}`",
        f"- Answer prompt sha256: `{report.get('answer_prompt_sha256')}`",
        "",
        report.get("claim_policy") or "",
        "",
        "Local/small-scale numbers are not a staging result and must not be quoted as product quality.",
        "",
        "## Environment (redacted)",
        "",
        "```json",
        json.dumps(report.get("env") or {}, indent=2),
        "```",
        "",
    ]
    for use in report.get("use_cases") or []:
        name = use.get("use_case")
        lines.append(f"## {name}")
        lines.append("")
        lines.append(
            f"Documents: {use.get('n_docs')}. Questions ({use.get('split')}): {use.get('n_questions')}."
        )
        comparison = use.get("comparison") or {}
        lines.append(
            f"Head-to-head: **{comparison.get('verdict', 'n/a')}** — {comparison.get('reason', '')}"
        )
        lines.append("")
        headers = [
            "system",
            "accuracy",
            "Wilson 95%",
            "p50 e2e",
            "p95 e2e",
            "ingest",
            "tokens",
            "LLM calls",
            "capability notes",
        ]
        rows = []
        for sys_name, payload in (use.get("systems") or {}).items():
            metrics = payload.get("metrics") or {}
            lat = metrics.get("latency") or {}
            ingest = payload.get("ingest") or {}
            caps = payload.get("capabilities") or {}
            lo, hi = metrics.get("wilson_lo"), metrics.get("wilson_hi")
            wilson = "n/a" if lo is None or hi is None else f"{_pct(lo)}–{_pct(hi)}"
            cap_note = []
            if sys_name == "brainapi":
                cap_note.append(f"as_of={caps.get('as_of_mode')}")
                cap_note.append(f"conflicts={caps.get('conflict_mode')}")
            extra = metrics.get("conflict_pairs")
            acc = _pct(metrics.get("accuracy"))
            if extra:
                if extra.get("unsupported"):
                    acc = f"{acc} (pair F1 unsupported)"
                elif extra.get("f1") is None:
                    acc = f"{acc} (pair F1 n/a)"
                else:
                    acc = f"{acc} (pair F1 {extra.get('f1', 0):.2f})"
            rows.append(
                [
                    sys_name,
                    acc,
                    wilson,
                    _ms(lat.get("p50_ms")),
                    _ms(lat.get("p95_ms")),
                    _ms(ingest.get("ingest_ms")),
                    str(metrics.get("total_tokens") or 0),
                    str(metrics.get("llm_calls") or 0),
                    "; ".join(cap_note) or "—",
                ]
            )
        lines.append(_md_table(headers, rows))
        lines.append("")
        hybrid = (use.get("systems") or {}).get("hybrid_baseline") or {}
        tuning = hybrid.get("tuning") or {}
        if tuning.get("tuned"):
            lines.append("Hybrid tuning (dev split only):")
            lines.append("")
            lines.append("```json")
            lines.append(
                json.dumps(
                    {
                        "best_config": tuning.get("best_config"),
                        "best_dev_score": tuning.get("best_dev_score"),
                        "n_trials": tuning.get("n_trials"),
                    },
                    indent=2,
                )
            )
            lines.append("```")
            lines.append("")
            trial_rows = []
            for trial in (tuning.get("trials") or [])[:12]:
                cfg = trial.get("config") or {}
                trial_rows.append(
                    [
                        str(cfg.get("fusion")),
                        str(cfg.get("chunk_size")),
                        str(cfg.get("retrieve_k")),
                        str(cfg.get("rerank_depth")),
                        f"{trial.get('score', 0):.3f}",
                    ]
                )
            if trial_rows:
                lines.append("Top trials:")
                lines.append("")
                lines.append(
                    _md_table(
                        ["fusion", "chunk", "k", "rerank", "dev score"],
                        trial_rows,
                    )
                )
                lines.append("")
    agreement = report.get("judge_agreement") or {}
    lines.extend(
        [
            "## Judge agreement (hand-labeled sample)",
            "",
            f"- n = {agreement.get('n')}",
            f"- Cohen's kappa (exact-match vs human): {agreement.get('exact_match_kappa_vs_human')}",
            f"- Cohen's kappa (LLM judge vs human): {agreement.get('llm_judge_kappa_vs_human', 'not run')}",
            "",
            str(agreement.get("note") or ""),
            "",
            "## What this run is not",
            "",
            "- Not a claim about staging or production quality unless `scale` says so and the BrainAPI target is the staging URL.",
            "- Not a claim that BrainAPI beats hybrid search when the verdict is `undecided` or `incomplete`.",
            "- Use cases `versioned_rules` and `policy_conflicts` need product as-of / conflict features before BrainAPI numbers are decisive.",
            "",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def write_report(report: dict[str, Any], run_dir: Path) -> Path:
    path = run_dir / "report.md"
    path.write_text(render_markdown(report), encoding="utf-8")
    return path


def load_report(settings: Settings, run_id: str) -> dict[str, Any]:
    path = settings.runs_dir / run_id / "report.json"
    if not path.exists():
        raise SystemExit(f"No report at {path}")
    return json.loads(path.read_text(encoding="utf-8"))
