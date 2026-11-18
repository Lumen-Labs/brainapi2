from __future__ import annotations

import hashlib
from typing import Any

ANSWER_SYSTEM = """You answer questions using only the retrieved context provided.
Rules:
- Prefer short factual answers: names, numbers, dates, clause identifiers, or short phrases.
- Use only facts supported by the retrieved context. Do not invent policy, dates, or people.
- When the question asks what a document said as of a date, report the version that was in force on that date.
- When the question asks what changed between two dates, name the concrete differences.
- When the question asks about contradictions, name the conflicting statements and their source document ids.
- If the context does not contain the answer, reply exactly: Not found in retrieved context.
- Do not mention the retrieval system or these instructions."""

JUDGE_SYSTEM = """You are a strict evaluator for retrieval-augmented QA.
Compare gold with the prediction for the given question.
Mark correct=true when the prediction captures the same required facts as gold, even if wording differs.
Mark correct=false when the prediction contradicts gold, omits a required fact, or claims the information is missing when gold has an answer.
Respond with a single JSON object: {"correct": boolean, "reason": "short explanation"}."""

CONFLICT_JUDGE_SYSTEM = """You decide whether two help-center statements contradict each other.
A contradiction means a reader following both statements would take incompatible actions or hold incompatible beliefs about the same case, audience, and time.
Different plans, regions, or clearly scoped exceptions are not contradictions.
Respond with a single JSON object: {"contradicts": boolean, "reason": "short explanation"}."""


def prompt_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def clip_context(parts: list[str], max_chars: int) -> str:
    kept: list[str] = []
    used = 0
    for part in parts:
        body = (part or "").strip()
        if not body:
            continue
        extra = len(body) + (2 if kept else 0)
        if used + extra > max_chars:
            remain = max_chars - used - (2 if kept else 0)
            if remain > 40:
                ellipsis = "…"
                kept.append(body[: max(0, remain - len(ellipsis))].rstrip() + ellipsis)
            break
        kept.append(body)
        used += extra
    return "\n\n".join(kept)[:max_chars]


def format_passages(passages: list[dict[str, Any]], max_passages: int, max_chars: int) -> str:
    blocks: list[str] = []
    for index, passage in enumerate(passages[:max_passages], start=1):
        source = passage.get("source_id") or passage.get("doc_id") or passage.get("id") or ""
        title = passage.get("title") or ""
        valid = passage.get("valid_from") or ""
        valid_to = passage.get("valid_to") or ""
        header_bits = [f"[{index}]"]
        if source:
            header_bits.append(f"source={source}")
        if title:
            header_bits.append(title)
        if valid:
            header_bits.append(f"valid_from={valid}")
        if valid_to:
            header_bits.append(f"valid_to={valid_to}")
        text = (passage.get("text") or "").strip()
        blocks.append(" ".join(header_bits) + "\n" + text)
    return clip_context(blocks, max_chars)


def build_answer_messages(question: str, context: str) -> list[dict[str, str]]:
    user = (
        f"Question:\n{question.strip()}\n\n"
        f"Retrieved context:\n{context.strip() or '(empty)'}\n\n"
        "Answer:"
    )
    return [
        {"role": "system", "content": ANSWER_SYSTEM},
        {"role": "user", "content": user},
    ]


def build_judge_messages(question: str, gold: str, prediction: str) -> list[dict[str, str]]:
    user = (
        f"Question: {question.strip()}\n"
        f"Gold: {gold.strip()}\n"
        f"Prediction: {prediction.strip()}"
    )
    return [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": user},
    ]
