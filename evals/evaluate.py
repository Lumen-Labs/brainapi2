from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evals.adapters.base import QueryResult, SystemAdapter
from evals.adapters.brainapi import (
    CONFLICT_ERROR_NOTES,
    CONFLICT_SKIP_NOTES,
    BrainAPIAdapter,
)
from evals.adapters.hybrid import HybridAdapter
from evals.baseline.dense import make_embedder
from evals.baseline.rerank import CrossEncoderReranker
from evals.baseline.tune import SMOKE_GRID, SOTA_TUNE_GRID, TUNE_GRID, tune_hybrid
from evals.config import Settings, embedding_dim_for
from evals.preflight import preflight_brainapi_model
from evals.datasets.loader import docs_for_run, load_judge_labels, load_use_case, pairs_for_run
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


def degraded_stages(value: Any) -> list[str]:
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if str(item).strip()]
    return []


def row_is_degraded(row: dict[str, Any] | None) -> bool:
    return bool(degraded_stages((row or {}).get("degraded")))


def _degraded_from_result(result: QueryResult) -> list[str]:
    raw = result.raw or {}
    stages = degraded_stages(raw.get("degraded"))
    if stages:
        return stages
    context = raw.get("context")
    if isinstance(context, dict):
        return degraded_stages(context.get("degraded"))
    return []


def _record(
    use_case: str,
    system: str,
    question: dict[str, Any],
    result: QueryResult,
    score: dict[str, Any],
) -> dict[str, Any]:
    degraded = _degraded_from_result(result)
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
        "as_of_mode": (result.raw or {}).get("as_of_mode"),
        "conflict_mode": (result.raw or {}).get("conflict_mode"),
        "conflict_mapping": (result.raw or {}).get("conflict_mapping"),
        "retrieve_ms": result.retrieve_ms,
        "answer_ms": result.answer_ms,
        "e2e_ms": result.e2e_ms,
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "total_tokens": result.total_tokens,
        "llm_calls": result.llm_calls,
        "http_calls": result.http_calls,
        **score,
        "degraded": degraded or None,
    }


def _score_one(
    question: dict[str, Any],
    result: QueryResult,
    *,
    gold_pairs: list[tuple[str, str]] | None = None,
    settings: Settings | None = None,
    use_judge: bool = False,
) -> dict[str, Any]:
    as_of_mode = (result.raw or {}).get("as_of_mode")
    if as_of_mode == "unsupported" and (
        str(question.get("task") or "") == "changelog" or question.get("date_to")
    ):
        return {
            "unsupported": True,
            "exact_match": False,
            "must_include": False,
            "token_f1": None,
            "correct": None,
            "scoring": "as_of_unsupported",
            "reason": "BrainAPI changelog retrieval has no supported range or history mode",
        }
    if question.get("task") == "conflict_audit":
        notes = result.notes or []
        gold = question.get("gold_pairs")
        if gold is None and gold_pairs is not None:
            gold = gold_pairs
        gold_t = [pair_key(str(a), str(b)) for a, b in (gold or [])]
        pred_t = [pair_key(str(a), str(b)) for a, b in result.conflict_pairs]
        error_note = next((item for item in notes if item in CONFLICT_ERROR_NOTES), None)
        if error_note:
            reasons = {
                "conflict_scan_error": "BrainAPI conflict scan failed",
                "conflict_scan_task_failed": "BrainAPI conflict scan task did not complete",
                "conflict_list_error": "BrainAPI conflict list failed after scan",
            }
            return {
                "unsupported": False,
                "error": reasons.get(error_note, error_note),
                "exact_match": False,
                "must_include": False,
                "token_f1": None,
                "correct": False,
                "scoring": "conflict_scan_error",
                "non_pair_leaks": 0,
                "n_gold": len(gold_t),
                "n_predicted": 0,
                "reason": reasons.get(error_note, error_note),
            }
        mapping_unsupported = any(item in CONFLICT_SKIP_NOTES for item in notes)
        if mapping_unsupported:
            pending = (
                "conflict_capability_pending" in notes
                or "conflict_scan_unavailable" in notes
            )
            return {
                "unsupported": True,
                "exact_match": False,
                "must_include": False,
                "token_f1": None,
                "correct": None,
                "scoring": "conflict_pairs_unsupported",
                "non_pair_leaks": 0,
                "n_gold": len(gold_t),
                "n_predicted": 0,
                "reason": (
                    "BrainAPI conflict scan/list is not available on this deployment"
                    if "conflict_scan_unavailable" in notes
                    else (
                        "BrainAPI conflict pair producer is not available on this deployment"
                        if pending
                        else "BrainAPI conflict identifiers could not be mapped to statement/doc gold ids"
                    )
                ),
            }
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


def _error_record(use_case: str, system: str, question: dict[str, Any], exc: Exception) -> dict[str, Any]:
    return {
        "use_case": use_case,
        "system": system,
        "question_id": question.get("id"),
        "split": question.get("split"),
        "task": question.get("task"),
        "question": question.get("question"),
        "gold": question.get("gold"),
        "error": str(exc),
        "correct": False,
    }


def _query_records(
    adapter: SystemAdapter,
    questions: list[dict[str, Any]],
    *,
    use_case: str,
    gold_pairs: list[tuple[str, str]],
    settings: Settings,
    use_judge: bool,
) -> list[dict[str, Any]]:
    pending: list[tuple[dict[str, Any], QueryResult | None, Exception | None]] = []
    for question in questions:
        try:
            pending.append((question, adapter.query(question), None))
        except Exception as exc:
            pending.append((question, None, exc))
    finalize = getattr(adapter, "finalize_conflict_mappings", None)
    if callable(finalize):
        finalize()
    records: list[dict[str, Any]] = []
    for question, result, exc in pending:
        if exc is not None or result is None:
            records.append(_error_record(use_case, adapter.name, question, exc or RuntimeError("query failed")))
            continue
        try:
            score = _score_one(
                question,
                result,
                gold_pairs=gold_pairs if question.get("task") == "conflict_audit" else None,
                settings=settings,
                use_judge=use_judge,
            )
            records.append(_record(use_case, adapter.name, question, result, score))
        except Exception as score_exc:
            records.append(_error_record(use_case, adapter.name, question, score_exc))
    return records


def _aggregate(
    records: list[dict[str, Any]], *, include_degraded: bool = False
) -> dict[str, Any]:
    n = len(records)
    scored = [row for row in records if row.get("error") is None]
    n_unsupported = sum(1 for row in records if row.get("unsupported") and row.get("error") is None)
    n_degraded = sum(
        1
        for row in records
        if row_is_degraded(row) and row.get("error") is None and not row.get("unsupported")
    )
    excluded = n_unsupported + (0 if include_degraded else n_degraded)
    denom = n - excluded

    def _counts_for_accuracy(row: dict[str, Any]) -> bool:
        if row.get("error") is not None or row.get("unsupported"):
            return False
        if not include_degraded and row_is_degraded(row):
            return False
        return True

    successes = sum(1 for row in records if row.get("correct") and _counts_for_accuracy(row))
    lo, hi = wilson_ci(successes, denom)
    f1_rows = [
        row
        for row in records
        if not row.get("unsupported") and (include_degraded or not row_is_degraded(row))
    ]
    return {
        "n": n,
        "n_errors": sum(1 for row in records if row.get("error")),
        "n_unsupported": n_unsupported,
        "n_degraded": n_degraded,
        "include_degraded": include_degraded,
        "accuracy": (successes / denom) if denom else None,
        "wilson_lo": lo,
        "wilson_hi": hi,
        "mean_token_f1": (
            sum(float(row.get("token_f1") or 0.0) for row in f1_rows) / len(f1_rows)
            if f1_rows
            else None
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


def _conflict_headline_unsupported(records: list[dict[str, Any]]) -> bool:
    return any(
        row.get("unsupported")
        or row.get("scoring") == "conflict_pairs_unsupported"
        or any(item in CONFLICT_SKIP_NOTES for item in (row.get("notes") or []))
        for row in records
    )


def _conflict_headline_error(records: list[dict[str, Any]]) -> bool:
    return any(
        row.get("scoring") == "conflict_scan_error"
        or any(item in CONFLICT_ERROR_NOTES for item in (row.get("notes") or []))
        for row in records
    )


def _conflict_headline_reason(records: list[dict[str, Any]]) -> str:
    if any("conflict_scan_unavailable" in (row.get("notes") or []) for row in records):
        for row in records:
            reason = str(row.get("reason") or "").strip()
            if reason:
                return reason
        return "BrainAPI conflict scan/list is not available on this deployment"
    pending = any(
        "conflict_capability_pending" in (row.get("notes") or []) for row in records
    )
    mapping = any(
        "conflict_mapping_unsupported" in (row.get("notes") or []) for row in records
    )
    if pending and not mapping:
        for row in records:
            reason = str(row.get("reason") or "").strip()
            if reason:
                return reason
        return "BrainAPI conflict pair producer is not available on this deployment"
    return "BrainAPI conflict identifiers could not be mapped to statement/doc gold ids"


def _conflict_headline(records: list[dict[str, Any]], gold_pairs: list[tuple[str, str]]) -> dict[str, Any]:
    if _conflict_headline_error(records):
        reason = next(
            (
                str(row.get("error") or row.get("reason") or "").strip()
                for row in records
                if row.get("error") or row.get("reason")
            ),
            "BrainAPI conflict scan failed",
        )
        return {
            "unsupported": False,
            "error": reason or "BrainAPI conflict scan failed",
            "f1": None,
            "precision": None,
            "recall": None,
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "n_predicted": 0,
            "n_gold": len(gold_pairs),
            "scoring": "conflict_scan_error",
        }
    if _conflict_headline_unsupported(records):
        return {
            "unsupported": True,
            "reason": _conflict_headline_reason(records),
            "f1": None,
            "precision": None,
            "recall": None,
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "n_predicted": 0,
            "n_gold": len(gold_pairs),
        }
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
    adapter = HybridAdapter(settings, answerer=answerer, smoke=smoke)
    tune_info: dict[str, Any] = {"tuned": False, "config": dict(adapter.config)}
    if tune and dev_questions:
        if smoke:
            grid = SMOKE_GRID
        elif settings.baseline_profile == "sota":
            grid = SOTA_TUNE_GRID
        else:
            grid = TUNE_GRID
        if smoke:
            embedder = make_embedder("hash", dim=256)
            cross = None
        else:
            try:
                embedder = make_embedder(
                    settings.embedding_provider,
                    api_key=settings.embedding_api_key,
                    model=settings.embedding_model,
                    base_url=settings.embedding_base_url,
                    dim=embedding_dim_for(settings.embedding_model),
                )
                cross = None
                if settings.reranker in {"cross-encoder", "ce", "cross_encoder"}:
                    cross = CrossEncoderReranker(settings.cross_encoder_model)
            except ImportError as exc:
                raise SystemExit(str(exc)) from exc
        result = tune_hybrid(
            docs,
            dev_questions,
            grid=grid,
            embedder=embedder,
            cross_encoder=cross,
        )
        adapter.config = dict(result["best_config"])
        if smoke:
            adapter.config["dense"] = "hash"
            adapter.config["reranker"] = "lexical"
            result["best_config"] = dict(adapter.config)
        adapter.index = result["index"].with_config(adapter.config)
        adapter.docs = list(docs)
        adapter._conflict_cache = None
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
    smoke_run = smoke or split == "smoke"
    docs = docs_for_run(bundle["corpus"], smoke=smoke_run)
    gold_pairs = pairs_for_run(bundle["gold_pairs"], docs, smoke=smoke_run)
    system_runs: dict[str, Any] = {}
    all_records: list[dict[str, Any]] = []
    tune_questions = (
        bundle["questions"]["smoke"] if smoke_run else bundle["questions"]["dev"]
    )

    if "hybrid" in systems:
        hybrid, tune_info = _build_hybrid(
            settings,
            docs,
            tune_questions,
            answerer=answerer,
            tune=tune,
            smoke=smoke_run,
        )
        records = _query_records(
            hybrid,
            questions,
            use_case=name,
            gold_pairs=gold_pairs,
            settings=settings,
            use_judge=use_judge,
        )
        all_records.extend(records)
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
        records = _query_records(
            adapter,
            questions,
            use_case=name,
            gold_pairs=gold_pairs,
            settings=settings,
            use_judge=use_judge,
        )
        all_records.extend(records)
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


REQUIRED_BRAINAPI_CAPS = {
    "versioned_rules": ("as_of",),
    "policy_conflicts": ("conflict_pairs",),
}


def _system_question_ids(records: list[dict[str, Any]], system: str) -> set[Any]:
    return {
        row.get("question_id")
        for row in records
        if row.get("system") == system
        and not row.get("unsupported")
        and not row_is_degraded(row)
    }


def compare_systems(
    system_runs: dict[str, Any],
    *,
    use_case: str | None = None,
    records: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if "hybrid_baseline" not in system_runs or "brainapi" not in system_runs:
        return {
            "verdict": "incomplete",
            "reason": "Both systems must complete the same items before a win is claimed.",
        }
    caps = (system_runs.get("brainapi") or {}).get("capabilities") or {}
    if not caps.get("conflict_error"):
        for cap in REQUIRED_BRAINAPI_CAPS.get(use_case or "", ()):
            if not caps.get(cap):
                return {
                    "verdict": "capability_pending",
                    "reason": (
                        f"BrainAPI {cap} is not available on this run; "
                        "not scored as a loss."
                    ),
                    "missing_capability": cap,
                }
    if use_case == "versioned_rules":
        mode = str(caps.get("as_of_mode") or "")
        if mode not in {"api_field", "api_schema"}:
            return {
                "verdict": "capability_pending",
                "reason": (
                    "BrainAPI native as-of is not available on this run; "
                    "include_history is not a date filter and is not scored as a win."
                ),
                "missing_capability": "as_of",
            }
        if records is not None and any(
            row.get("system") == "brainapi"
            and (
                row.get("as_of_mode") == "unsupported"
                or row.get("scoring") == "as_of_unsupported"
            )
            for row in records
        ):
            return {
                "verdict": "capability_pending",
                "reason": (
                    "BrainAPI changelog retrieval had no supported range mode; "
                    "not scored as a win."
                ),
                "missing_capability": "as_of",
            }
    if records is not None:
        hybrid_ids = _system_question_ids(records, "hybrid_baseline")
        brain_ids = _system_question_ids(records, "brainapi")
        if hybrid_ids != brain_ids:
            return {
                "verdict": "incomplete",
                "reason": "Systems did not score the same question ids.",
                "hybrid_n": len(hybrid_ids),
                "brainapi_n": len(brain_ids),
            }
    h = system_runs["hybrid_baseline"]["metrics"]
    b = system_runs["brainapi"]["metrics"]
    h_n = (
        int(h.get("n") or 0)
        - int(h.get("n_unsupported") or 0)
        - int(h.get("n_degraded") or 0)
    )
    b_n = (
        int(b.get("n") or 0)
        - int(b.get("n_unsupported") or 0)
        - int(b.get("n_degraded") or 0)
    )
    if h_n != b_n:
        return {
            "verdict": "incomplete",
            "reason": "Systems did not score the same number of items.",
            "hybrid_n": h_n,
            "brainapi_n": b_n,
        }
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
    include_brainapi = "brainapi" in systems and bool(settings.brainpat_token)
    smoke_run = smoke or split == "smoke"
    if include_brainapi and smoke_run:
        skip_enrichment = True
        llm_preflight: dict[str, Any] = {
            "ok": True,
            "mode": "smoke_skip_enrichment",
            "llm_model": settings.llm_model,
            "note": "CI smoke skips BrainAPI graph enrichment and does not call server LLMs.",
        }
    elif include_brainapi:
        if skip_enrichment:
            raise SystemExit(
                "skip-enrichment is CI-smoke only. Staging/real BrainAPI comparisons "
                "must run ingest enrichment (graph extraction) so the arm matches the product."
            )
        llm_preflight = preflight_brainapi_model(settings)
        skip_enrichment = False
    else:
        llm_preflight = {
            "ok": True,
            "mode": "brainapi_not_attempted",
            "llm_model": settings.llm_model,
        }
    rid, run_dir = ensure_run_dir(settings, run_id)
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
        comparison = compare_systems(
            result["systems"],
            use_case=name,
            records=result["records"],
        )
        result["comparison"] = comparison
        rows.append({k: v for k, v in result.items() if k != "records"})
        all_records.extend(result["records"])
        write_jsonl(run_dir / f"{name}.jsonl", result["records"])

    agreement = judge_agreement(settings, settings.datasets_dir)
    if use_judge and settings.llm_api_key:
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
        "llm_model": settings.llm_model,
        "answer_model": settings.llm_model,
        "judge_model": settings.llm_model,
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
        "brainapi_skip_enrichment": skip_enrichment,
        "brainapi_llm_preflight": llm_preflight,
        "systems_requested": systems,
        "baseline_tier": "smoke_hash" if smoke_run else "strong",
        "baseline_profile": "smoke_hash" if smoke_run else settings.baseline_profile,
        "embedding_model": (
            "hash" if smoke_run else settings.embedding_model
        ),
        "reranker_model": (
            "lexical" if smoke_run else settings.cross_encoder_model
        ),
    }
    write_jsonl(run_dir / "answers.jsonl", all_records)
    (run_dir / "report.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return payload
