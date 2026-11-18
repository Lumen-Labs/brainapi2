from __future__ import annotations

import math
import re
from collections.abc import Iterable, Sequence
from typing import Any

_WS_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w\s]+", re.UNICODE)


def normalize_text(value: str) -> str:
    text = (value or "").strip().lower()
    text = text.replace("“", '"').replace("”", '"').replace("’", "'")
    text = _PUNCT_RE.sub(" ", text)
    return _WS_RE.sub(" ", text).strip()


def tokenize(value: str) -> list[str]:
    return [tok for tok in normalize_text(value).split(" ") if tok]


def exact_match(prediction: str, gold: str, aliases: Sequence[str] | None = None) -> bool:
    pred = normalize_text(prediction)
    candidates = [gold, *(aliases or [])]
    return any(pred == normalize_text(item) for item in candidates if item)


def must_include(prediction: str, phrases: Sequence[str]) -> bool:
    pred = f" {normalize_text(prediction)} "
    if not phrases:
        return False
    needles = [normalize_text(phrase) for phrase in phrases]
    return all(f" {needle} " in pred for needle in needles if needle)


def token_f1(prediction: str, gold: str) -> float:
    pred_toks = tokenize(prediction)
    gold_toks = tokenize(gold)
    if not pred_toks and not gold_toks:
        return 1.0
    if not pred_toks or not gold_toks:
        return 0.0
    overlap = 0
    gold_counts: dict[str, int] = {}
    for tok in gold_toks:
        gold_counts[tok] = gold_counts.get(tok, 0) + 1
    for tok in pred_toks:
        if gold_counts.get(tok, 0) > 0:
            overlap += 1
            gold_counts[tok] -= 1
    precision = overlap / len(pred_toks)
    recall = overlap / len(gold_toks)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def percentile(values: Iterable[float], p: float) -> float | None:
    ordered = sorted(float(v) for v in values)
    if not ordered:
        return None
    if len(ordered) == 1:
        return ordered[0]
    k = (len(ordered) - 1) * (p / 100.0)
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return ordered[int(k)]
    return ordered[lo] * (hi - k) + ordered[hi] * (k - lo)


def mean(values: Iterable[float]) -> float | None:
    vals = [float(v) for v in values]
    if not vals:
        return None
    return sum(vals) / len(vals)


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float | None, float | None]:
    if n <= 0:
        return None, None
    p = successes / n
    denom = 1.0 + (z * z) / n
    center = (p + (z * z) / (2.0 * n)) / denom
    margin = z * math.sqrt((p * (1.0 - p) / n) + (z * z) / (4.0 * n * n)) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def pair_key(left: str, right: str) -> tuple[str, str]:
    a, b = str(left), str(right)
    return (a, b) if a <= b else (b, a)


def conflict_prf(
    predicted: Sequence[tuple[str, str]],
    gold: Sequence[tuple[str, str]],
) -> dict[str, float | int]:
    pred_set = {pair_key(a, b) for a, b in predicted if a and b and a != b}
    gold_set = {pair_key(a, b) for a, b in gold if a and b and a != b}
    tp = len(pred_set & gold_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    )
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "n_predicted": len(pred_set),
        "n_gold": len(gold_set),
    }


def cohens_kappa(labels_a: Sequence[bool], labels_b: Sequence[bool]) -> float | None:
    if len(labels_a) != len(labels_b) or not labels_a:
        return None
    n = len(labels_a)
    agree = sum(1 for a, b in zip(labels_a, labels_b) if bool(a) == bool(b))
    p_o = agree / n
    a_true = sum(1 for a in labels_a if a) / n
    b_true = sum(1 for b in labels_b if b) / n
    p_e = a_true * b_true + (1 - a_true) * (1 - b_true)
    if p_e == 1:
        return 1.0
    return (p_o - p_e) / (1 - p_e)


def mcnemar_exact(flipped_to_correct: int, flipped_to_wrong: int) -> float | None:
    n = flipped_to_correct + flipped_to_wrong
    if n == 0:
        return 1.0
    k = min(flipped_to_correct, flipped_to_wrong)
    p = 0.0
    for i in range(k + 1):
        p += math.comb(n, i)
    return min(1.0, 2.0 * p / (2**n))


def latency_summary(values: Iterable[float]) -> dict[str, float | None]:
    vals = [float(v) for v in values]
    return {
        "n": len(vals),
        "mean_ms": mean(vals),
        "p50_ms": percentile(vals, 50),
        "p95_ms": percentile(vals, 95),
    }


def score_prediction(
    prediction: str,
    question: dict[str, Any],
) -> dict[str, Any]:
    gold = str(question.get("gold") or "")
    aliases = list(question.get("aliases") or [])
    required = list(question.get("must_include") or [])
    mode = str(question.get("scoring") or "exact_match")
    em = exact_match(prediction, gold, aliases)
    included = must_include(prediction, required) if required else em
    f1 = token_f1(prediction, gold) if gold else 0.0
    if mode == "must_include":
        correct = included
    elif mode == "token_f1":
        correct = f1 >= 0.7
    elif mode == "llm_judge":
        correct = None
    else:
        correct = em or included
    return {
        "exact_match": em,
        "must_include": included,
        "token_f1": f1,
        "correct": correct,
        "scoring": mode,
    }
