from __future__ import annotations

from typing import Any

REQUIRED_DOC_FIELDS = ("id", "text")
REQUIRED_QUESTION_FIELDS = ("id", "question", "split")


def validate_doc(doc: dict[str, Any], path: str) -> None:
    missing = [field for field in REQUIRED_DOC_FIELDS if not doc.get(field)]
    if missing:
        raise ValueError(f"{path}: document missing {missing}: {doc.get('id')}")


def validate_question(question: dict[str, Any], path: str) -> None:
    missing = [field for field in REQUIRED_QUESTION_FIELDS if not question.get(field)]
    if missing:
        raise ValueError(f"{path}: question missing {missing}: {question.get('id')}")
    split = question.get("split")
    if split not in {"smoke", "dev", "test"}:
        raise ValueError(f"{path}: bad split {split!r} on {question.get('id')}")
