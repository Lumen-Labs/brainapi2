from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any

from evals.config import Settings
from evals.prompts import CONFLICT_JUDGE_SYSTEM, build_judge_messages


@dataclass
class JudgeResult:
    correct: bool
    reason: str
    raw: str
    model: str
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


def _client(settings: Settings):
    from openai import AzureOpenAI, OpenAI

    settings.require_judge_llm()
    if settings.judge_azure_endpoint:
        return AzureOpenAI(
            api_key=settings.judge_api_key,
            azure_endpoint=settings.judge_azure_endpoint,
            api_version=settings.judge_azure_api_version,
        )
    kwargs: dict[str, Any] = {"api_key": settings.judge_api_key}
    if settings.judge_base_url:
        kwargs["base_url"] = settings.judge_base_url
    return OpenAI(**kwargs)


def _parse_bool_payload(raw: str, key: str) -> tuple[bool, str]:
    text = raw.strip()
    try:
        data = json.loads(text)
        return bool(data.get(key)), str(data.get("reason", ""))
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
                return bool(data.get(key)), str(data.get("reason", ""))
            except json.JSONDecodeError:
                pass
    lowered = text.lower()
    if f'"{key}": true' in lowered or f"{key}: true" in lowered:
        return True, text
    if f'"{key}": false' in lowered or f"{key}: false" in lowered:
        return False, text
    return False, f"unparseable judge output: {text[:200]}"


def judge_answer(
    settings: Settings,
    question: str,
    gold: str,
    prediction: str,
) -> JudgeResult:
    client = _client(settings)
    messages = build_judge_messages(question, gold, prediction)
    started = time.perf_counter()
    response = client.chat.completions.create(
        model=settings.judge_model,
        messages=messages,
        temperature=0,
        response_format={"type": "json_object"},
    )
    latency_ms = (time.perf_counter() - started) * 1000
    raw = (response.choices[0].message.content or "").strip()
    correct, reason = _parse_bool_payload(raw, "correct")
    usage = response.usage
    return JudgeResult(
        correct=correct,
        reason=reason,
        raw=raw,
        model=getattr(response, "model", None) or settings.judge_model,
        latency_ms=latency_ms,
        prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0) if usage else 0,
        completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0) if usage else 0,
        total_tokens=int(getattr(usage, "total_tokens", 0) or 0) if usage else 0,
    )


def judge_conflict_pair(
    settings: Settings,
    left: str,
    right: str,
) -> JudgeResult:
    client = _client(settings)
    messages = [
        {"role": "system", "content": CONFLICT_JUDGE_SYSTEM},
        {
            "role": "user",
            "content": f"Statement A:\n{left}\n\nStatement B:\n{right}",
        },
    ]
    started = time.perf_counter()
    response = client.chat.completions.create(
        model=settings.judge_model,
        messages=messages,
        temperature=0,
        response_format={"type": "json_object"},
    )
    latency_ms = (time.perf_counter() - started) * 1000
    raw = (response.choices[0].message.content or "").strip()
    correct, reason = _parse_bool_payload(raw, "contradicts")
    usage = response.usage
    return JudgeResult(
        correct=correct,
        reason=reason,
        raw=raw,
        model=getattr(response, "model", None) or settings.judge_model,
        latency_ms=latency_ms,
        prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0) if usage else 0,
        completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0) if usage else 0,
        total_tokens=int(getattr(usage, "total_tokens", 0) or 0) if usage else 0,
    )
