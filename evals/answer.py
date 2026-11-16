from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from evals.config import Settings
from evals.prompts import build_answer_messages, format_passages


@dataclass
class AnswerResult:
    text: str
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    model: str
    calls: int


def _openai_client(settings: Settings):
    from openai import AzureOpenAI, OpenAI

    if settings.judge_azure_endpoint and not settings.llm_base_url:
        return AzureOpenAI(
            api_key=settings.judge_api_key,
            azure_endpoint=settings.judge_azure_endpoint,
            api_version=settings.judge_azure_api_version,
        )
    kwargs: dict[str, Any] = {"api_key": settings.openai_api_key}
    if settings.llm_base_url:
        kwargs["base_url"] = settings.llm_base_url
    return OpenAI(**kwargs)


def pack_context(passages: list[dict[str, Any]], settings: Settings) -> str:
    budget = settings.prompt_budget
    return format_passages(
        passages,
        max_passages=int(budget["max_passages"]),
        max_chars=int(budget["max_chars"]),
    )


def llm_answer(
    settings: Settings,
    question: str,
    passages: list[dict[str, Any]],
) -> AnswerResult:
    settings.require_llm()
    client = _openai_client(settings)
    context = pack_context(passages, settings)
    messages = build_answer_messages(question, context)
    started = time.perf_counter()
    response = client.chat.completions.create(
        model=settings.answer_model,
        messages=messages,
        temperature=float(settings.prompt_budget["temperature"]),
        max_tokens=int(settings.prompt_budget["max_answer_tokens"]),
    )
    latency_ms = (time.perf_counter() - started) * 1000
    text = (response.choices[0].message.content or "").strip()
    usage = response.usage
    return AnswerResult(
        text=text,
        latency_ms=latency_ms,
        prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0) if usage else 0,
        completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0) if usage else 0,
        total_tokens=int(getattr(usage, "total_tokens", 0) or 0) if usage else 0,
        model=getattr(response, "model", None) or settings.answer_model,
        calls=1,
    )


def extractive_answer(question: dict[str, Any], passages: list[dict[str, Any]]) -> AnswerResult:
    from evals.baseline.tune import extractive_answer as extract

    text = extract(question, passages)
    return AnswerResult(
        text=text,
        latency_ms=0.0,
        prompt_tokens=0,
        completion_tokens=0,
        total_tokens=0,
        model="extractive",
        calls=0,
    )
