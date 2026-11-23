from __future__ import annotations

from typing import Any, Callable, Iterable

DEFAULT_MAX_DEGRADED_RATE = 0.05

RowPredicate = Callable[[dict[str, Any]], bool]


def degraded_stages(value: Any) -> list[str]:
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if str(item).strip()]
    return []


def row_is_degraded(row: dict[str, Any] | None) -> bool:
    return bool(degraded_stages((row or {}).get("degraded")))


def _as_rate(n_degraded: int, n_total: int) -> float | None:
    if n_total <= 0:
        return None
    return n_degraded / n_total


def binary_strict_accuracy(
    rows: Iterable[dict[str, Any]],
    *,
    is_degraded: RowPredicate = row_is_degraded,
    is_correct: RowPredicate | None = None,
    correct_key: str = "judge_correct",
) -> float | None:
    items = list(rows)
    if not items:
        return None
    correct = is_correct or (lambda row: bool(row.get(correct_key)))
    successes = sum(1 for row in items if not is_degraded(row) and correct(row))
    return successes / len(items)


def score_degraded_rows(
    rows: Iterable[dict[str, Any]],
    *,
    is_degraded: RowPredicate = row_is_degraded,
    is_correct: RowPredicate | None = None,
    correct_key: str = "judge_correct",
    max_degraded_rate: float = DEFAULT_MAX_DEGRADED_RATE,
) -> dict[str, Any]:
    items = list(rows)
    n_total = len(items)
    n_degraded = sum(1 for row in items if is_degraded(row))
    rate = _as_rate(n_degraded, n_total)
    return {
        "n_total": n_total,
        "n_degraded": n_degraded,
        "degraded_rate": rate,
        "max_degraded_rate": max_degraded_rate,
        "strict_accuracy": binary_strict_accuracy(
            items,
            is_degraded=is_degraded,
            is_correct=is_correct,
            correct_key=correct_key,
        ),
        "exceeded_max_degraded_rate": bool(
            rate is not None and rate > max_degraded_rate
        ),
    }


def zero_degraded_scores(
    rows: Iterable[dict[str, Any]],
    *,
    is_degraded: RowPredicate = row_is_degraded,
    score_keys: tuple[str, ...] = ("llm_judge_score", "tau_norm"),
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        if not is_degraded(row):
            out.append(row)
            continue
        cloned = dict(row)
        for key in score_keys:
            cloned[key] = 0.0
        out.append(cloned)
    return out


def add_max_degraded_rate_argument(parser: Any) -> None:
    parser.add_argument(
        "--max-degraded-rate",
        type=float,
        default=DEFAULT_MAX_DEGRADED_RATE,
        metavar="RATE",
        help=(
            "Exit non-zero when degraded retrieval rows exceed this fraction "
            f"of unique scored answers (default: {DEFAULT_MAX_DEGRADED_RATE:g})."
        ),
    )


def report_exit_code(report: dict[str, Any]) -> int:
    if report.get("status") == "failed":
        return 1
    integrity = report.get("integrity") or {}
    if integrity.get("exceeded_max_degraded_rate"):
        return 1
    metrics = report.get("metrics") or {}
    if metrics.get("exceeded_max_degraded_rate"):
        return 1
    for use in report.get("use_cases") or []:
        for payload in (use.get("systems") or {}).values():
            use_metrics = payload.get("metrics") or {}
            if use_metrics.get("exceeded_max_degraded_rate"):
                return 1
    return 0
