from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evals.adapters.base import QueryResult, SystemAdapter
from evals.adapters.brainapi import BrainAPIAdapter
from evals.adapters.hybrid import HybridAdapter
from evals.baseline.tune import SMOKE_GRID, TUNE_GRID, tune_hybrid
from evals.config import Settings
from evals.datasets.loader import load_judge_labels, load_use_case
from evals.judge import judge_answer
from evals.metrics import (
    cohens_kappa,
    conflict_prf,
    exact_match,
    latency_summary,
    pair_key,
    score_prediction,
    wilson_ci,
)
from evals.prompts import ANSWER_SYSTEM, prompt_sha256


def ensure_run_dir(settings: Settings, run_id: str | None = None) -> tuple[str, Path]:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    rid = run_id or f"eval-{stamp}"
    run_dir = settings.runs_dir / rid
    run_dir.mkdir(parents=True, exist_ok=True)
    return rid, run_dir


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _record(
    use_case: str,
    system: str,
    question: dict[str, Any],
    result: QueryResult,
    score: dict[str, Any],
) -> dict[str, Any]:
    return {
        "use_case": use_case,
        "system": system,
        "question_id": question.get("id"),
        "split": question.get("split"),
        "task": question.get("task"),
        "question": question.get("question"),
        "gold": question.get("gold"),
        "prediction": result.answer,
        "conflict_pairs": [list(pair) for pair in result.conflict_pairs],
        "notes": result.notes,
        "retrieve_ms": result.retrieve_ms,
        "answer_ms": result.answer_ms,
        "e2e_ms": result.e2e_ms,
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "total_tokens": result.total_tokens,
        "llm_calls": result.llm_calls,
        "http_calls": result.http_calls,
        **score,
    }


def _score_one(
    question: dict[str, Any],
    result: QueryResult,
    *,
    gold_pairs: list[tuple[str, str]] | None = None,
    settings: Settings | None = None,
    use_judge: bool = False,
) -> dict[str, Any]:
    if question.get("task") == "conflict_audit":
        gold = question.get("gold_pairs")
        if gold is None and gold_pairs is not None:
            gold = gold_pairs
        gold_t = [pair_key(str(a), str(b)) for a, b in (gold or [])]
        pred_t = [pair_key(str(a), str(b)) for a, b in result.conflict_pairs]
        prf = conflict_prf(pred_t, gold_t)
        non_pairs = [pair_key(str(a), str(b)) for a, b in (question.get("non_pairs") or [])]
        leaked = [pair for pair in pred_t if pair in set(non_pairs)]
        if gold_t:
            found = set(pred_t)
            recall_q = sum(1 for pair in gold_t if pair in found) / len(gold_t)
            correct = recall_q >= 1.0 and len(leaked) == 0
            prf["question_recall"] = recall_q
        else:
            correct = len(leaked) == 0
        return {
            **prf,
            "exact_match": False,
            "must_include": False,
            "token_f1": prf["f1"],
            "correct": bool(correct),
            "scoring": "conflict_pairs",
            "non_pair_leaks": len(leaked),
        }
    scored = score_prediction(result.answer, question)
    if scored["correct"] is None or (
        use_judge and settings and str(question.get("scoring") or "") == "llm_judge"
    ):
        if use_judge and settings:
            judged = judge_answer(
                settings,
                str(question.get("question") or ""),
                str(question.get("gold") or ""),
                result.answer,
            )
            scored["correct"] = judged.correct
            scored["judge_reason"] = judged.reason
            scored["judge_model"] = judged.model
            scored["judge_tokens"] = judged.total_tokens
            scored["scoring"] = "llm_judge"
        else:
            scored["correct"] = bool(scored["must_include"] or scored["exact_match"])
    return scored


def _aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    scored = [row for row in records if row.get("error") is None]
    n = len(scored)
    successes = sum(1 for row in scored if row.get("correct"))
    lo, hi = wilson_ci(successes, n)
    return {
        "n": n,
        "n_errors": sum(1 for row in records if row.get("error")),
        "accuracy": (successes / n) if n else None,
        "wilson_lo": lo,
        "wilson_hi": hi,
        "mean_token_f1": (
            sum(float(row.get("token_f1") or 0.0) for row in scored) / n if n else None
        ),
        "latency": latency_summary(float(row["e2e_ms"]) for row in scored if row.get("e2e_ms") is not None),
        "retrieve_latency": latency_summary(
            float(row["retrieve_ms"]) for row in scored if row.get("retrieve_ms") is not None
        ),
        "prompt_tokens": sum(int(row.get("prompt_tokens") or 0) for row in records),
        "completion_tokens": sum(int(row.get("completion_tokens") or 0) for row in records),
        "total_tokens": sum(int(row.get("total_tokens") or 0) for row in records),
        "llm_calls": sum(int(row.get("llm_calls") or 0) for row in records),
        "http_calls": sum(int(row.get("http_calls") or 0) for row in records),
    }


def _conflict_headline(records: list[dict[str, Any]], gold_pairs: list[tuple[str, str]]) -> dict[str, Any]:
    pred: list[tuple[str, str]] = []
    for row in records:
        for pair in row.get("conflict_pairs") or []:
            if isinstance(pair, (list, tuple)) and len(pair) >= 2:
                pred.append(pair_key(str(pair[0]), str(pair[1])))
    unique = list(dict.fromkeys(pred))
    return conflict_prf(unique, [pair_key(a, b) for a, b in gold_pairs])


def judge_agreement(settings: Settings, datasets_dir: Path | None = None) -> dict[str, Any]:
    rows = load_judge_labels(datasets_dir)
    human = [bool(row.get("human_correct")) for row in rows]
    em = [
        exact_match(str(row.get("prediction") or ""), str(row.get("gold") or ""))
        or False
        for row in rows
    ]
    # Exact-match on these rows is a proxy when no LLM judge is invoked.
    kappa_em = cohens_kappa(human, em)
    payload: dict[str, Any] = {
        "n": len(rows),
        "human_positive": sum(human),
        "exact_match_kappa_vs_human": kappa_em,
        "note": (
            "Hand-labeled sample in datasets/judge_agreement.jsonl. "
            "Kappa vs exact-match is reported even without an LLM judge. "
            "Pass --judge to also score the LLM judge on this sample."
        ),
        "items": [
            {
                "id": row.get("id"),
                "human_correct": bool(row.get("human_correct")),
                "exact_match": em[idx],
            }
            for idx, row in enumerate(rows)
        ],
    }
    return payload


def _build_hybrid(
    settings: Settings,
    docs: list[dict[str, Any]],
    dev_questions: list[dict[str, Any]],
    *,
    answerer: str,
    tune: bool,
    smoke: bool,
) -> tuple[HybridAdapter, dict[str, Any]]:
    adapter = HybridAdapter(settings, answerer=answerer)
    tune_info: dict[str, Any] = {"tuned": False, "config": dict(adapter.config)}
    if tune and dev_questions:
        grid = SMOKE_GRID if smoke else TUNE_GRID
        result = tune_hybrid(docs, dev_questions, grid=grid)
        adapter.config = dict(result["best_config"])
        adapter.index = result["index"]
        adapter.docs = list(docs)
        adapter.ingest_stats = {
            "docs": len(docs),
            "chunks": len(result["index"].chunks),
            "ingest_ms": result["index"].build_ms,
            "cost": dict(result["index"].cost),
            "config": dict(adapter.config),
        }
        tune_info = {
            "tuned": True,
            "split": "dev",
            "best_config": result["best_config"],
            "best_dev_score": result["best_score"],
            "n_trials": result["n_trials"],
            "trials": result["trials"],
            "grid": result["grid"],
        }
        return adapter, tune_info
    adapter.ingest(docs)
    return adapter, tune_info


def evaluate_use_case(
    name: str,
    settings: Settings,
    *,
    split: str,
    systems: list[str],
    answerer: str,
    tune: bool,
    smoke: bool,
    use_judge: bool,
    skip_enrichment: bool,
    include_brainapi: bool,
) -> dict[str, Any]:
    bundle = load_use_case(name, settings.datasets_dir)
    questions = bundle["questions"].get(split) or []
    if not questions:
        raise SystemExit(f"No questions for {name} split={split}")
    docs = bundle["corpus"]
    gold_pairs = bundle["gold_pairs"]
    system_runs: dict[str, Any] = {}
    all_records: list[dict[str, Any]] = []

    if "hybrid" in systems:
        hybrid, tune_info = _build_hybrid(
            settings,
            docs,
            bundle["questions"]["dev"],
            answerer=answerer,
            tune=tune,
            smoke=smoke,
        )
        records = []
        for question in questions:
            result = hybrid.query(question)
            score = _score_one(
                question,
                result,
                gold_pairs=gold_pairs if question.get("task") == "conflict_audit" else None,
                settings=settings,
                use_judge=use_judge,
            )
            rec = _record(name, hybrid.name, question, result, score)
            records.append(rec)
            all_records.append(rec)
        headline = _aggregate(records)
        if gold_pairs:
            headline["conflict_pairs"] = _conflict_headline(records, gold_pairs)
        system_runs["hybrid_baseline"] = {
            "capabilities": hybrid.capabilities,
            "ingest": hybrid.ingest_stats,
            "tuning": tune_info,
            "metrics": headline,
        }

    if "brainapi" in systems and include_brainapi:
        adapter = BrainAPIAdapter(
            settings,
            brain_id=settings.brain_id_for(name),
            answerer=answerer,
            skip_enrichment=skip_enrichment,
        )
        ingest = adapter.ingest(docs)
        records = []
        for question in questions:
            try:
                result = adapter.query(question)
                score = _score_one(
                    question,
                    result,
                    gold_pairs=gold_pairs if question.get("task") == "conflict_audit" else None,
                    settings=settings,
                    use_judge=use_judge,
                )
                rec = _record(name, adapter.name, question, result, score)
            except Exception as exc:
                rec = {
                    "use_case": name,
                    "system": adapter.name,
                    "question_id": question.get("id"),
                    "error": str(exc),
                    "correct": False,
                }
            records.append(rec)
            all_records.append(rec)
        headline = _aggregate(records)
        if gold_pairs:
            headline["conflict_pairs"] = _conflict_headline(records, gold_pairs)
        system_runs["brainapi"] = {
            "capabilities": adapter.capabilities,
            "ingest": {
                "docs": ingest.get("docs"),
                "ingest_ms": ingest.get("ingest_ms"),
                "skip_enrichment": ingest.get("skip_enrichment"),
                "brain_id": ingest.get("brain_id"),
                "http_calls": ingest.get("http_calls"),
            },
            "metrics": headline,
        }

    return {
        "use_case": name,
        "split": split,
        "n_docs": len(docs),
        "n_questions": len(questions),
        "systems": system_runs,
        "records": all_records,
    }


def compare_systems(system_runs: dict[str, Any]) -> dict[str, Any]:
    names = list(system_runs)
    if "hybrid_baseline" not in system_runs or "brainapi" not in system_runs:
        return {
            "verdict": "incomplete",
            "reason": "Both systems must complete the same items before a win is claimed.",
        }
    h = system_runs["hybrid_baseline"]["metrics"]
    b = system_runs["brainapi"]["metrics"]
    if h.get("accuracy") is None or b.get("accuracy") is None:
        return {"verdict": "undecided", "reason": "accuracy unavailable"}
    h_lo, h_hi = h.get("wilson_lo"), h.get("wilson_hi")
    b_lo, b_hi = b.get("wilson_lo"), b.get("wilson_hi")
    overlap = not (
        h_hi is not None
        and b_lo is not None
        and b_lo > h_hi
        or (b_hi is not None and h_lo is not None and h_lo > b_hi)
    )
    if overlap:
        return {
            "verdict": "undecided",
            "reason": "Wilson 95% intervals overlap; no claim.",
            "hybrid_accuracy": h.get("accuracy"),
            "brainapi_accuracy": b.get("accuracy"),
        }
    winner = "brainapi" if (b.get("accuracy") or 0) > (h.get("accuracy") or 0) else "hybrid_baseline"
    return {
        "verdict": winner,
        "reason": "Wilson 95% intervals do not overlap on this split.",
        "hybrid_accuracy": h.get("accuracy"),
        "brainapi_accuracy": b.get("accuracy"),
    }


def evaluate(
    settings: Settings,
    *,
    use_cases: list[str],
    split: str,
    systems: list[str],
    run_id: str | None,
    answerer: str,
    tune: bool,
    smoke: bool,
    use_judge: bool,
    skip_enrichment: bool,
) -> dict[str, Any]:
    rid, run_dir = ensure_run_dir(settings, run_id)
    include_brainapi = "brainapi" in systems and bool(settings.brainpat_token)
    rows = []
    all_records: list[dict[str, Any]] = []
    for name in use_cases:
        result = evaluate_use_case(
            name,
            settings,
            split=split,
            systems=systems,
            answerer=answerer,
            tune=tune,
            smoke=smoke,
            use_judge=use_judge,
            skip_enrichment=skip_enrichment,
            include_brainapi=include_brainapi,
        )
        comparison = compare_systems(result["systems"])
        result["comparison"] = comparison
        rows.append({k: v for k, v in result.items() if k != "records"})
        all_records.extend(result["records"])
        write_jsonl(run_dir / f"{name}.jsonl", result["records"])

    agreement = judge_agreement(settings, settings.datasets_dir)
    if use_judge and settings.judge_api_key:
        human = []
        judged = []
        for item in load_judge_labels(settings.datasets_dir):
            verdict = judge_answer(
                settings,
                str(item.get("question") or ""),
                str(item.get("gold") or ""),
                str(item.get("prediction") or ""),
            )
            human.append(bool(item.get("human_correct")))
            judged.append(verdict.correct)
        agreement["llm_judge_kappa_vs_human"] = cohens_kappa(human, judged)

    payload = {
        "schema_version": settings.schema_version,
        "run_id": rid,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scale": settings.scale if not smoke else "local/small-scale",
        "split": split,
        "smoke": smoke,
        "answerer": answerer,
        "prompt_budget": dict(settings.prompt_budget),
        "answer_prompt_sha256": prompt_sha256(ANSWER_SYSTEM),
        "env": settings.redacted_env(),
        "claim_policy": (
            "No claim that BrainAPI beats hybrid search unless the numbers show it "
            "on the same scored items with non-overlapping Wilson intervals. "
            "Capability-pending cells are not losses."
        ),
        "use_cases": rows,
        "judge_agreement": agreement,
        "brainapi_attempted": include_brainapi,
        "systems_requested": systems,
    }
    write_jsonl(run_dir / "answers.jsonl", all_records)
    (run_dir / "report.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return payload
