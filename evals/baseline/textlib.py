from __future__ import annotations

import re

TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?", re.UNICODE)
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n+")
STOP = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "of",
        "to",
        "in",
        "on",
        "for",
        "with",
        "from",
        "this",
        "that",
        "is",
        "are",
        "was",
        "were",
        "be",
        "as",
        "by",
        "at",
        "it",
    }
)


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall((text or "").lower())


def content_tokens(text: str) -> list[str]:
    return [tok for tok in tokenize(text) if tok not in STOP and len(tok) > 1]


def sentences(text: str) -> list[str]:
    parts = [part.strip() for part in SENTENCE_RE.split(text or "") if part.strip()]
    return parts or ([text.strip()] if text and text.strip() else [])
