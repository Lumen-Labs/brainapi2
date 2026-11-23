from __future__ import annotations

import json
import re
from typing import Any, Optional
from urllib.parse import urlparse

from src.core.saving.conflicts import normalize_predicate

_PREDICATE_PATTERNS: tuple[tuple[str, str, str, str], ...] = (
    (r"^(?P<s>.+?)\s+lives?\s+in\s+(?P<o>.+)$", "LIVES_IN", "PERSON", "PLACE"),
    (r"^(?P<s>.+?)\s+lives?\s+at\s+(?P<o>.+)$", "LIVES_AT", "PERSON", "PLACE"),
    (r"^(?P<s>.+?)\s+works?\s+at\s+(?P<o>.+)$", "WORKS_AT", "PERSON", "ORG"),
    (r"^(?P<s>.+?)\s+works?\s+for\s+(?P<o>.+)$", "EMPLOYED_BY", "PERSON", "ORG"),
    (r"^(?P<s>.+?)\s+is\s+headquartered\s+in\s+(?P<o>.+)$", "HEADQUARTERED_IN", "ORG", "PLACE"),
    (r"^(?P<s>.+?)\s+is\s+located\s+in\s+(?P<o>.+)$", "LOCATED_IN", "ENTITY", "PLACE"),
    (r"^(?P<s>.+?)\s+has\s+status\s+(?P<o>.+)$", "HAS_STATUS", "ENTITY", "VALUE"),
    (r"^(?P<s>.+?)\s+requires?\s+(?P<o>.+)$", "REQUIRES", "ENTITY", "VALUE"),
    (r"^(?P<s>.+?)\s+(?:is|are|was|were)\s+(?P<o>.+)$", "IS", "ENTITY", "VALUE"),
)

_ENTITY_TYPES = {
    "PERSON",
    "ORG",
    "PLACE",
    "ENTITY",
    "VALUE",
    "FACT",
    "EVENT",
    "CLAUSE",
}

_LLM_UNSET = object()


class ParsedFact:
    __slots__ = ("text", "subject", "predicate", "object", "subject_type", "object_type")

    def __init__(
        self,
        text: str,
        subject: str,
        predicate: str,
        object: str,
        subject_type: str,
        object_type: str,
    ):
        self.text = text
        self.subject = subject
        self.predicate = predicate
        self.object = object
        self.subject_type = subject_type
        self.object_type = object_type


def _clean_phrase(value: str) -> str:
    text = re.sub(r"\s+", " ", (value or "").strip())
    return text.rstrip(" .;,:")


def _normalize_entity_type(value: Any, fallback: str) -> str:
    text = str(value or "").strip().upper().replace(" ", "_")
    if text in _ENTITY_TYPES:
        return text
    if "PERSON" in text or text in {"PEOPLE", "HUMAN"}:
        return "PERSON"
    if text in {"ORGANIZATION", "COMPANY", "ORG"}:
        return "ORG"
    if text in {"LOCATION", "CITY", "COUNTRY", "PLACE"}:
        return "PLACE"
    return fallback


def parse_fact_text_regex(text: str) -> ParsedFact:
    raw = _clean_phrase(text)
    if not raw:
        raise ValueError("fact text is required")
    for pattern, predicate, subject_type, object_type in _PREDICATE_PATTERNS:
        match = re.match(pattern, raw, re.IGNORECASE)
        if not match:
            continue
        subject = _clean_phrase(match.group("s"))
        obj = _clean_phrase(match.group("o"))
        if subject and obj:
            return ParsedFact(
                raw,
                subject,
                predicate,
                obj,
                subject_type,
                object_type,
            )
    return ParsedFact(raw, raw, "STATES", "recorded", "FACT", "VALUE")


def _configured_llm() -> Any | None:
    try:
        from src.core.instances import llm_small_adapter

        return llm_small_adapter
    except Exception:
        return None


def _parse_llm_payload(payload: Any) -> dict[str, Any] | None:
    if isinstance(payload, list) and payload:
        payload = payload[0]
    if isinstance(payload, str):
        text = payload.strip().strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()
        try:
            payload = json.loads(text)
        except Exception:
            return None
    if not isinstance(payload, dict):
        return None
    nested = payload.get("fact")
    if isinstance(nested, dict):
        payload = nested
    return payload


def extract_fact_with_llm(text: str, llm: Any) -> ParsedFact | None:
    raw = _clean_phrase(text)
    if not raw or llm is None:
        return None
    prompt = (
        "Extract one knowledge-graph triple from the statement. "
        "The statement may be in any language. Reply with JSON only: "
        '{"subject":"...","predicate":"SCREAMING_SNAKE","object":"...",'
        '"subject_type":"PERSON|ORG|PLACE|ENTITY|VALUE",'
        '"object_type":"PERSON|ORG|PLACE|ENTITY|VALUE"} '
        "Predicate is a relationship name, not a sentence. "
        f"Statement: {raw}"
    )
    try:
        generate = getattr(llm, "generate_json", None)
        if callable(generate):
            parsed = generate(prompt, max_new_tokens=200)
        else:
            generate_text = getattr(llm, "generate_text", None)
            if not callable(generate_text):
                return None
            parsed = generate_text(prompt, max_new_tokens=200)
    except Exception:
        return None
    payload = _parse_llm_payload(parsed)
    if not payload:
        return None
    subject = _clean_phrase(str(payload.get("subject") or ""))
    obj = _clean_phrase(str(payload.get("object") or payload.get("obj") or ""))
    predicate = normalize_predicate(str(payload.get("predicate") or ""))
    if not subject or not obj or not predicate:
        return None
    return ParsedFact(
        raw,
        subject,
        predicate,
        obj,
        _normalize_entity_type(payload.get("subject_type"), "ENTITY"),
        _normalize_entity_type(payload.get("object_type"), "ENTITY"),
    )


def parse_fact_text(text: str, llm: Any = _LLM_UNSET) -> ParsedFact:
    regex = parse_fact_text_regex(text)
    extractor = _configured_llm() if llm is _LLM_UNSET else llm
    if extractor is None:
        return regex
    extracted = extract_fact_with_llm(regex.text, extractor)
    return extracted or regex


def parse_source(source: str | None, source_key: str | None = None) -> dict[str, Optional[str]]:
    text = (source or "").strip()
    payload: dict[str, Optional[str]] = {
        "doc": None,
        "url": None,
        "source_key": (source_key or "").strip() or None,
    }
    if not text:
        return payload
    parsed = urlparse(text)
    if parsed.scheme in {"http", "https"} and parsed.netloc:
        payload["url"] = text
        payload["doc"] = parsed.netloc + (parsed.path or "")
        return payload
    payload["doc"] = text
    return payload


def fact_statement(subject: str, predicate: str, obj: str, fallback: str = "") -> str:
    pred = normalize_predicate(predicate).replace("_", " ").strip().lower()
    if subject and pred and obj:
        return f"{subject} {pred} {obj}"
    return fallback or " ".join(part for part in (subject, predicate, obj) if part)


def tokenize(text: str) -> set[str]:
    return {part for part in re.findall(r"[a-z0-9]+", (text or "").lower()) if len(part) > 1}


def lexical_score(query: str, candidate: str) -> float:
    left = tokenize(query)
    right = tokenize(candidate)
    if not left or not right:
        return 0.0
    overlap = left & right
    if not overlap:
        return 0.0
    return len(overlap) / max(len(left), 1)
