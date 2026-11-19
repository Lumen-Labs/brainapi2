from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from evals.baseline.dates import require_document_source_timestamp
from evals.config import DATASETS_DIR, USE_CASES
from evals.datasets.schema import validate_doc, validate_question

USE_CASE_DIRS = {
    "versioned_rules": "versioned_rules",
    "policy_conflicts": "policy_conflicts",
    "expert_knowhow": "expert_knowhow",
    "decision_precedent": "decision_precedent",
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            raw = line.strip()
            if not raw or raw.startswith("#"):
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: expected object")
            rows.append(row)
    return rows


def use_case_dir(name: str, datasets_dir: Path | None = None) -> Path:
    if name not in USE_CASE_DIRS:
        raise SystemExit(f"Unknown use case {name!r}. Choose from {USE_CASES}")
    return (datasets_dir or DATASETS_DIR) / USE_CASE_DIRS[name]


def load_corpus(name: str, datasets_dir: Path | None = None) -> list[dict[str, Any]]:
    path = use_case_dir(name, datasets_dir) / "corpus.jsonl"
    rows = _read_jsonl(path)
    for row in rows:
        validate_doc(row, str(path))
        if name == "versioned_rules":
            require_document_source_timestamp(row, str(path))
    return rows


def load_questions(
    name: str,
    *,
    split: str | None = None,
    datasets_dir: Path | None = None,
) -> list[dict[str, Any]]:
    path = use_case_dir(name, datasets_dir) / "questions.jsonl"
    rows = _read_jsonl(path)
    for row in rows:
        validate_question(row, str(path))
    if split:
        rows = [row for row in rows if row.get("split") == split]
    return rows


def load_gold_pair_rows(
    name: str, datasets_dir: Path | None = None
) -> list[dict[str, Any]]:
    path = use_case_dir(name, datasets_dir) / "gold_pairs.jsonl"
    if not path.exists():
        return []
    return _read_jsonl(path)


def load_gold_pairs(
    name: str, datasets_dir: Path | None = None
) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for row in load_gold_pair_rows(name, datasets_dir):
        left = str(row.get("left") or "")
        right = str(row.get("right") or "")
        if left and right:
            pairs.append((left, right))
    return pairs


def docs_for_run(docs: list[dict[str, Any]], *, smoke: bool) -> list[dict[str, Any]]:
    if not smoke:
        return list(docs)
    tagged = [doc for doc in docs if doc.get("ci_smoke")]
    return tagged or list(docs)


def pairs_for_run(
    pairs: list[tuple[str, str]],
    docs: list[dict[str, Any]],
    *,
    smoke: bool,
) -> list[tuple[str, str]]:
    if not smoke:
        return list(pairs)
    allowed = {str(doc.get("statement_id") or doc.get("id") or "") for doc in docs}
    return [(left, right) for left, right in pairs if left in allowed and right in allowed]


def load_use_case(name: str, datasets_dir: Path | None = None) -> dict[str, Any]:
    directory = use_case_dir(name, datasets_dir)
    readme = directory / "README.md"
    return {
        "name": name,
        "dir": directory,
        "readme": readme.read_text(encoding="utf-8") if readme.exists() else "",
        "corpus": load_corpus(name, datasets_dir),
        "questions": {
            "all": load_questions(name, datasets_dir=datasets_dir),
            "smoke": load_questions(name, split="smoke", datasets_dir=datasets_dir),
            "dev": load_questions(name, split="dev", datasets_dir=datasets_dir),
            "test": load_questions(name, split="test", datasets_dir=datasets_dir),
        },
        "gold_pairs": load_gold_pairs(name, datasets_dir),
    }


def load_judge_labels(datasets_dir: Path | None = None) -> list[dict[str, Any]]:
    path = (datasets_dir or DATASETS_DIR) / "judge_agreement.jsonl"
    if not path.exists():
        return []
    return _read_jsonl(path)
