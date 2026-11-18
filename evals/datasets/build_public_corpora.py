#!/usr/bin/env python3
"""Build scaled eval corpora from public FAR/eCFR, help-center, and runbook sources.

Regenerate committed JSONL with:

    PYTHONPATH=. python3 -m evals.datasets.build_public_corpora

Network is required for eCFR XML. GitHub Docs, GSA-TTS handbook, and
prometheus-operator runbooks are read from local clones (override with flags).
Gold answers for versioned rules are derived from section-level diffs; policy
pairs mix naturally occurring numeric clashes with documented seeded mutations;
know-how gold is extracted from runbook headings and first mitigation/diagnosis
steps. See each dataset README and GOLD.md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
DATASETS = Path(__file__).resolve().parent
CACHE = DATASETS / ".cache"
DEFAULT_CLONES = {
    "github_docs": Path("/tmp/eval-sources/gh-docs"),
    "tts_handbook": Path("/tmp/eval-sources/tts-handbook"),
    "po_runbooks": Path("/tmp/eval-sources/po-runbooks"),
}

FAR_DATES = [
    "2018-01-03",
    "2019-01-03",
    "2020-01-02",
    "2021-01-04",
    "2022-01-03",
    "2023-01-03",
    "2024-01-02",
    "2025-01-02",
    "2026-10-28",
]
FAR_PARTS = ["2", "4", "13", "15", "16", "19", "22", "32", "36", "42"]
ECFR = "https://www.ecfr.gov/api/versioner/v1/full/{date}/title-48.xml"
USER_AGENT = "brainapi-evals/0.2 (public-corpus builder; no secrets)"
SKIP_TAGS = {"AUTH", "CITA", "SOURCE", "EDNOTE", "HED"}
FRONTMATTER_RE = re.compile(r"^---\r?\n.*?\r?\n---\r?\n", re.S)
LIQUID_RE = re.compile(r"{[{%].*?[%}]}", re.S)
HTML_RE = re.compile(r"<[^>]+>")
MD_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")
HEADING_RE = re.compile(r"^#{1,6}\s+", re.M)
FACT_RE = re.compile(
    r"(?:\$[\d,]+(?:\.\d+)?"
    r"|\b\d{1,4}(?:,\d{3})*(?:\.\d+)?\s*(?:percent|%|calendar days|business days|"
    r"days|day|months|month|years|year|hours|hour|minutes|minute|characters|"
    r"character|weeks|week)\b"
    r"|\bNet\s+\d+\b)",
    re.I,
)
SENT_RE = re.compile(r"(?<=[.!?])\s+")
SECTION_ID_RE = re.compile(r"^\d{1,2}\.\d+")
WS_RE = re.compile(r"\s+")


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _norm_space(text: str) -> str:
    return WS_RE.sub(" ", text).strip()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw or raw.startswith("#"):
            continue
        rows.append(json.loads(raw))
    return rows


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            key = str(row.get("id") or json.dumps(row, sort_keys=True))
            if key in seen:
                continue
            seen.add(key)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _parse_iso(value: str) -> date:
    return date.fromisoformat(value[:10])


def _day_before(value: str) -> str:
    return (_parse_iso(value) - timedelta(days=1)).isoformat()


def _midpoint(start: str, end: str | None) -> str:
    left = _parse_iso(start)
    right = _parse_iso(end) if end else left + timedelta(days=30)
    if right <= left:
        return start
    mid = left + (right - left) / 2
    return mid.isoformat()


def strip_markup(raw: str) -> str:
    text = FRONTMATTER_RE.sub("", raw)
    replacements = {
        "variables.product.prodname_dotcom": "GitHub",
        "variables.product.github": "GitHub",
        "variables.product.prodname_enterprise": "GitHub Enterprise",
    }
    for key, value in replacements.items():
        text = text.replace("{% data " + key + " %}", value)
        text = text.replace("{{ " + key + " }}", value)
    text = LIQUID_RE.sub(" ", text)
    text = HTML_RE.sub(" ", text)
    text = MD_LINK_RE.sub(r"\1", text)
    text = HEADING_RE.sub("", text)
    text = text.replace("`", "")
    return _norm_space(text)


def sentences(text: str) -> list[str]:
    parts = [part.strip() for part in SENT_RE.split(text) if part.strip()]
    return [part for part in parts if len(part) > 20]


def facts_in(text: str) -> list[str]:
    found = []
    for match in FACT_RE.finditer(text):
        span = _norm_space(match.group(0)).strip(" ,;:.")
        if SECTION_ID_RE.match(span) or len(span) < 2:
            continue
        if span.lower() not in {item.lower() for item in found}:
            found.append(span)
    return found


def fetch_far_xml(
    snapshot: str, part: str, cache_dir: Path, skip_fetch: bool = False
) -> Path | None:
    dest = cache_dir / f"title48-part{part}-{snapshot}.xml"
    if dest.exists() and dest.stat().st_size > 500:
        return dest
    if skip_fetch:
        return None
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        import httpx
    except ImportError as exc:
        raise SystemExit("httpx is required to fetch eCFR XML") from exc
    url = ECFR.format(date=snapshot)
    headers = {
        "User-Agent": USER_AGENT,
        "Accept-Encoding": "gzip, deflate",
    }
    with httpx.Client(timeout=90.0, follow_redirects=True, headers=headers) as client:
        response = client.get(url, params={"chapter": "1", "part": part})
    if response.status_code != 200 or len(response.content) < 500:
        print(f"skip FAR part {part} {snapshot} status={response.status_code} bytes={len(response.content)}")
        return None
    dest.write_bytes(response.content)
    return dest


def parse_far_sections(path: Path) -> list[dict[str, str]]:
    try:
        root = ET.fromstring(path.read_text(encoding="utf-8"))
    except ET.ParseError as exc:
        print(f"xml parse failed {path}: {exc}")
        return []
    rows: list[dict[str, str]] = []
    for el in root.iter():
        if not (el.tag.startswith("DIV") and el.attrib.get("TYPE") == "SECTION"):
            continue
        clause = (el.attrib.get("N") or "").strip()
        if not clause:
            continue
        head = _norm_space("".join((el.find("HEAD").itertext() if el.find("HEAD") is not None else [])))
        chunks: list[str] = []
        seen: set[str] = set()
        for child in el.iter():
            if child.tag in SKIP_TAGS:
                continue
            if child.tag in {"P", "FP", "HEAD"}:
                text = _norm_space("".join(child.itertext()))
                if text and text not in seen:
                    seen.add(text)
                    chunks.append(text)
        body = _norm_space(" ".join(chunks))
        if len(body) < 80:
            continue
        rows.append(
            {
                "clause_id": f"FAR {clause}",
                "title": head or f"FAR {clause}",
                "text": body[:3500],
            }
        )
    return rows


def collapse_versions(
    snapshots: dict[str, list[tuple[str, str, str]]],
) -> list[dict[str, Any]]:
    docs: list[dict[str, Any]] = []
    for clause_id, items in snapshots.items():
        items = sorted(items, key=lambda row: row[0])
        collapsed: list[tuple[str, str, str]] = []
        for snap, title, text in items:
            if collapsed and collapsed[-1][2] == text:
                continue
            collapsed.append((snap, title, text))
        if len(collapsed) < 2:
            continue
        for idx, (snap, title, text) in enumerate(collapsed):
            valid_to = _day_before(collapsed[idx + 1][0]) if idx + 1 < len(collapsed) else None
            docs.append(
                {
                    "id": f"far-{clause_id.split()[-1]}-v{idx+1}",
                    "clause_id": clause_id,
                    "title": title,
                    "version": snap,
                    "valid_from": snap,
                    "valid_to": valid_to,
                    "text": text,
                    "source": "eCFR Title 48 Chapter 1",
                    "source_url": (
                        f"https://www.ecfr.gov/on/{snap}/title-48/chapter-1/"
                        f"part-{clause_id.split()[-1].split('.')[0]}"
                    ),
                    "license": "U.S. government work, 17 U.S.C. § 105",
                    "ci_smoke": False,
                }
            )
    return docs


def build_versioned_questions(docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_clause: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for doc in docs:
        by_clause[str(doc["clause_id"])].append(doc)
    as_of_rows: list[dict[str, Any]] = []
    change_rows: list[dict[str, Any]] = []
    for clause_id, versions in sorted(by_clause.items()):
        versions = sorted(versions, key=lambda row: str(row["valid_from"]))
        fact_sets = [facts_in(str(row["text"])) for row in versions]
        for idx, doc in enumerate(versions):
            others = set()
            for jdx, facts in enumerate(fact_sets):
                if jdx != idx:
                    others.update(item.lower() for item in facts)
            distinctive = [item for item in fact_sets[idx] if item.lower() not in others]
            if not distinctive:
                continue
            gold = distinctive[0]
            as_of = _midpoint(str(doc["valid_from"]), doc.get("valid_to"))
            as_of_rows.append(
                {
                    "id": f"vr-far-asof-{doc['id']}",
                    "split": "pending",
                    "task": "as_of",
                    "as_of": as_of,
                    "clause_id": clause_id,
                    "question": (
                        f"As of {as_of}, what figure does {clause_id} "
                        f"({doc.get('title')}) specify that is unique to the version "
                        f"then in force?"
                    ),
                    "gold": gold,
                    "must_include": [gold],
                    "scoring": "must_include",
                    "gold_method": "diff_unique_numeric_span",
                    "source_doc": doc["id"],
                }
            )
        for idx in range(len(versions) - 1):
            left, right = versions[idx], versions[idx + 1]
            left_facts = {item.lower(): item for item in fact_sets[idx]}
            right_facts = {item.lower(): item for item in fact_sets[idx + 1]}
            removed = [left_facts[key] for key in left_facts if key not in right_facts]
            added = [right_facts[key] for key in right_facts if key not in left_facts]
            if not removed or not added:
                left_sents = set(sentences(str(left["text"])))
                right_sents = set(sentences(str(right["text"])))
                dropped = sorted(left_sents - right_sents)
                gained = sorted(right_sents - left_sents)
                if not dropped or not gained:
                    continue
                old_span = dropped[0][:80]
                new_span = gained[0][:80]
                must = [old_span[:40], new_span[:40]]
            else:
                old_span = removed[0]
                new_span = added[0]
                must = [old_span, new_span]
            a, b = must[0].strip().lower(), must[1].strip().lower()
            if not a or not b or a == b or a in b or b in a:
                continue
            date_from = _midpoint(str(left["valid_from"]), left.get("valid_to"))
            date_to = _midpoint(str(right["valid_from"]), right.get("valid_to"))
            change_rows.append(
                {
                    "id": f"vr-far-chg-{left['id']}-{right['id']}",
                    "split": "pending",
                    "task": "changelog",
                    "date_from": date_from,
                    "date_to": date_to,
                    "clause_id": clause_id,
                    "question": (
                        f"What changed in {clause_id} between {date_from} and {date_to}?"
                    ),
                    "gold": f"{old_span} became {new_span}",
                    "must_include": must,
                    "scoring": "must_include",
                    "gold_method": "consecutive_version_diff",
                    "source_doc": right["id"],
                }
            )
    # Prefer as-of items (they are the core as-of task), then changelogs.
    rows = as_of_rows + change_rows
    return rows


def split_assign(rows: list[dict[str, Any]], *, n_dev: int, n_test: int) -> list[dict[str, Any]]:
    assigned: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        clone = dict(row)
        if idx < n_dev:
            clone["split"] = "dev"
        elif idx < n_dev + n_test:
            clone["split"] = "test"
        else:
            break
        assigned.append(clone)
    return assigned


def split_fill_test(
    rows: list[dict[str, Any]],
    *,
    n_dev: int,
    min_test: int,
    max_test: int = 180,
) -> list[dict[str, Any]]:
    if len(rows) < n_dev + min_test:
        raise SystemExit(
            f"Need at least {n_dev + min_test} generated questions, got {len(rows)}"
        )
    n_test = min(max_test, len(rows) - n_dev)
    assigned: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        if idx >= n_dev + n_test:
            break
        clone = dict(row)
        clone["split"] = "dev" if idx < n_dev else "test"
        assigned.append(clone)
    return assigned


SMOKE_DOC_PREFIXES = ("famr-", "hc-", "sre-", "lab-")
SMOKE_QUESTION_PREFIXES = ("vr-smoke", "pc-smoke", "ek-smoke")
SMOKE_PAIR_LEFTS = {
    "stmt-refund-14",
    "stmt-retain-30",
    "stmt-pw-8",
    "stmt-2fa-optional",
}


def load_smoke_docs(path: Path) -> list[dict[str, Any]]:
    rows = []
    for row in _read_jsonl(path):
        doc_id = str(row.get("id") or "")
        if row.get("ci_smoke") and doc_id.startswith(SMOKE_DOC_PREFIXES):
            row["ci_smoke"] = True
            rows.append(row)
        elif doc_id.startswith(SMOKE_DOC_PREFIXES) and not str(row.get("source") or "").startswith(
            ("eCFR", "github", "tts", "prometheus")
        ):
            row["ci_smoke"] = True
            rows.append(row)
    return rows


def load_smoke_questions(path: Path) -> list[dict[str, Any]]:
    return [
        row
        for row in _read_jsonl(path)
        if row.get("split") == "smoke"
        and str(row.get("id") or "").startswith(SMOKE_QUESTION_PREFIXES)
    ]


def load_smoke_pairs(path: Path) -> list[dict[str, Any]]:
    rows = []
    for row in _read_jsonl(path):
        if row.get("left") in SMOKE_PAIR_LEFTS:
            row["ci_smoke"] = True
            rows.append(row)
    return rows


def build_far(
    cache_dir: Path, *, skip_fetch: bool = False
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    jobs: list[tuple[str, str]] = [(snap, part) for snap in FAR_DATES for part in FAR_PARTS]
    paths: dict[tuple[str, str], Path] = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        futs = {
            pool.submit(fetch_far_xml, snap, part, cache_dir, skip_fetch): (snap, part)
            for snap, part in jobs
        }
        for fut in as_completed(futs):
            snap, part = futs[fut]
            path = fut.result()
            if path is not None:
                paths[(snap, part)] = path
    snapshots: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    fetch_log = []
    for (snap, part), path in sorted(paths.items()):
        sections = parse_far_sections(path)
        fetch_log.append(
            {
                "date": snap,
                "part": part,
                "path": str(path),
                "n_sections": len(sections),
                "sha256_16": _sha(path.read_text(encoding="utf-8")),
            }
        )
        for section in sections:
            snapshots[section["clause_id"]].append((snap, section["title"], section["text"]))
    docs = collapse_versions(snapshots)
    questions = build_versioned_questions(docs)
    meta = {
        "fetched_at": _utc_now(),
        "n_xml": len(paths),
        "n_changing_clauses": len({doc["clause_id"] for doc in docs}),
        "n_docs": len(docs),
        "n_candidate_questions": len(questions),
        "fetch_log": fetch_log,
    }
    return docs, questions, meta


def iter_md(root: Path, *, min_chars: int = 200) -> list[tuple[Path, str, str]]:
    rows: list[tuple[Path, str, str]] = []
    if not root.exists():
        return rows
    for path in sorted(root.rglob("*.md")):
        name = path.name.lower()
        if name in {"readme.md", "contributing.md", "license.md", "security.md", "index.md", "_index.md"}:
            continue
        raw = path.read_text(encoding="utf-8", errors="ignore")
        text = strip_markup(raw)
        if len(text) < min_chars:
            continue
        title = path.stem.replace("-", " ")
        fm = FRONTMATTER_RE.search(raw)
        if fm:
            m = re.search(r"^title:\s*[\"']?(.+?)[\"']?\s*$", fm.group(0), re.M)
            if m:
                title = m.group(1).strip()
        rows.append((path, title, text[:4000]))
    return rows


def paragraphs_with_facts(text: str) -> list[str]:
    parts = [part.strip() for part in re.split(r"\n{2,}", text) if part.strip()]
    if len(parts) <= 1:
        parts = sentences(text)
    kept: list[str] = []
    for part in parts:
        clean = _norm_space(part)
        if len(clean) < 40 or len(clean) > 700:
            continue
        if facts_in(clean) or re.search(r"\b(required|optional|must|may not)\b", clean, re.I):
            kept.append(clean)
    return kept


def mutate_statement(text: str) -> tuple[str, str] | None:
    header_m = re.match(r"^(\[stmt-[^\]]+\]\s*)", text)
    header = header_m.group(1) if header_m else ""
    body = text[len(header) :]
    fact = facts_in(body)
    if fact:
        original = fact[0]
        digits = re.search(r"[\d.]+", original)
        if not digits:
            return None
        number = digits.group(0)
        try:
            value = float(number.replace(",", ""))
        except ValueError:
            return None
        if value.is_integer():
            nxt = int(value) + (7 if int(value) < 20 else max(3, int(value) // 2))
            mutated_num = str(nxt)
        else:
            mutated_num = str(round(value * 1.5, 1))
        mutated_fact = original[: digits.start()] + mutated_num + original[digits.end() :]
        idx = body.find(original)
        if idx < 0:
            return None
        mutated_body = body[:idx] + mutated_fact + body[idx + len(original) :]
        return header + mutated_body, f"{original} vs {mutated_fact}"
    if re.search(r"\brequired\b", body, re.I):
        mutated_body = re.sub(r"\brequired\b", "optional", body, count=1, flags=re.I)
        return header + mutated_body, "required vs optional"
    if re.search(r"\boptional\b", body, re.I):
        mutated_body = re.sub(r"\boptional\b", "required", body, count=1, flags=re.I)
        return header + mutated_body, "optional vs required"
    for lone in re.finditer(r"\b(\d{1,3})\b", body):
        number = lone.group(1)
        if int(number) >= 1900 or int(number) == 0:
            continue
        start, end = lone.span(1)
        ctx = body[max(0, start - 8) : end + 3]
        if re.search(r"CC0|Commons", ctx, re.I):
            continue
        nxt = str(int(number) + 5)
        mutated_body = body[:start] + nxt + body[end:]
        return header + mutated_body, f"{number} vs {nxt}"
    return None


GENERIC_TOPIC_TOKENS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "of",
    "for",
    "to",
    "in",
    "on",
    "your",
    "our",
    "with",
    "from",
    "plan",
    "plans",
    "standard",
    "enterprise",
    "github",
    "tts",
    "gsa",
    "handbook",
    "guide",
    "policy",
    "about",
    "faq",
    "docs",
    "documentation",
    "changes",
    "intro",
    "using",
    "how",
    "what",
    "when",
    "why",
    "overview",
    "index",
}


def topic_tokens(title: str) -> set[str]:
    return {
        tok
        for tok in re.findall(r"[a-z0-9]+", title.lower())
        if tok not in GENERIC_TOPIC_TOKENS and len(tok) > 2
    }


def fact_families(facts: Iterable[str]) -> set[str]:
    families: set[str] = set()
    for item in facts:
        text = item.lower()
        if re.search(r"\bdays?\b", text):
            families.add("day")
        if re.search(r"\bweeks?\b", text):
            families.add("week")
        if re.search(r"\bmonths?\b", text):
            families.add("month")
        if re.search(r"\byears?\b", text):
            families.add("year")
        if re.search(r"\bhours?\b", text):
            families.add("hour")
        if re.search(r"\bminutes?\b", text):
            families.add("minute")
        if "$" in text or "dollar" in text:
            families.add("money")
        if "percent" in text or "%" in text:
            families.add("percent")
        if "character" in text:
            families.add("length")
        if re.search(r"\bnet\s+\d+", text):
            families.add("payment_terms")
    return families


def is_natural_clash(left: dict[str, Any], right: dict[str, Any]) -> bool:
    if left.get("source") != right.get("source"):
        return False
    overlap = topic_tokens(str(left.get("title") or "")) & topic_tokens(str(right.get("title") or ""))
    if not overlap:
        return False
    left_facts = {item.lower() for item in facts_in(str(left.get("text") or ""))}
    right_facts = {item.lower() for item in facts_in(str(right.get("text") or ""))}
    if not left_facts or not right_facts or left_facts == right_facts or left_facts & right_facts:
        return False
    only_left = left_facts - right_facts
    only_right = right_facts - left_facts
    return bool(fact_families(only_left) & fact_families(only_right))


def build_policy(
    github_root: Path,
    tts_root: Path,
    existing_docs: list[dict[str, Any]],
    existing_pairs: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    sources: list[tuple[str, Path, str, str]] = []
    gh_files = iter_md(github_root / "content") if github_root.exists() else []
    tts_files = []
    if tts_root.exists():
        for rel in (
            "pages/travel-and-leave",
            "pages/general-information-and-resources/tech-policies",
            "pages/training-and-development",
            "pages/tools",
            "pages/getting-started",
        ):
            tts_files.extend(iter_md(tts_root / rel))
    for path, title, text in gh_files:
        sources.append(("github_docs", path, title, text))
    for path, title, text in tts_files:
        sources.append(("tts_handbook", path, title, text))

    originals: list[dict[str, Any]] = []
    for origin, path, title, text in sources:
        paras = paragraphs_with_facts(text)
        for idx, para in enumerate(paras[:6]):
            sid = f"stmt-{origin[:2]}-{_sha(str(path) + para)[:10]}"
            license_name = (
                "CC-BY-4.0 (GitHub Docs)"
                if origin == "github_docs"
                else "U.S. government work / CC0 (GSA-TTS handbook)"
            )
            originals.append(
                {
                    "id": f"doc-{sid}",
                    "statement_id": sid,
                    "title": title,
                    "text": f"[{sid}] {para}",
                    "source": origin,
                    "source_path": str(path),
                    "license": license_name,
                    "ci_smoke": False,
                    "seeded": False,
                }
            )
            if len(originals) >= 260:
                break
        if len(originals) >= 260:
            break

    gold_pairs: list[dict[str, Any]] = [dict(row, ci_smoke=row.get("ci_smoke", True)) for row in existing_pairs]
    docs = list(existing_docs)
    seeded_docs: list[dict[str, Any]] = []
    for doc in originals:
        mutated = mutate_statement(doc["text"])
        if mutated is None:
            continue
        new_text, rationale = mutated
        if new_text == doc["text"]:
            continue
        if "CC0" in doc["text"] and rationale.startswith("0 vs"):
            continue
        seed_id = f"{doc['statement_id']}-seed"
        seeded_docs.append(
            {
                "id": f"doc-{seed_id}",
                "statement_id": seed_id,
                "title": f"{doc['title']} (stale copy)",
                "text": new_text.replace(f"[{doc['statement_id']}]", f"[{seed_id}]", 1),
                "source": doc["source"],
                "source_path": doc.get("source_path"),
                "license": doc["license"],
                "ci_smoke": False,
                "seeded": True,
                "seed_of": doc["statement_id"],
                "seed_rationale": rationale,
            }
        )
        gold_pairs.append(
            {
                "left": doc["statement_id"],
                "right": seed_id,
                "topic": re.sub(r"[^a-z0-9]+", "_", str(doc["title"]).lower())[:40],
                "seeded": True,
                "rationale": rationale,
                "ci_smoke": False,
            }
        )
        if len(seeded_docs) >= 200:
            break

    # Naturally occurring clashes among originals (same topic tokens, same unit family).
    natural = 0
    for idx, left in enumerate(originals):
        for right in originals[idx + 1 :]:
            if not is_natural_clash(left, right):
                continue
            overlap = topic_tokens(str(left["title"])) & topic_tokens(str(right["title"]))
            lfacts = {item.lower() for item in facts_in(left["text"])}
            rfacts = {item.lower() for item in facts_in(right["text"])}
            gold_pairs.append(
                {
                    "left": left["statement_id"],
                    "right": right["statement_id"],
                    "topic": "|".join(sorted(overlap)[:4]),
                    "seeded": False,
                    "rationale": f"numeric clash {sorted(lfacts)[:2]} vs {sorted(rfacts)[:2]}",
                    "ci_smoke": False,
                }
            )
            natural += 1
            if natural >= 20:
                break
        if natural >= 20:
            break

    # Plan-scoped near-misses (must not be gold).
    near_miss_questions: list[dict[str, Any]] = []
    for doc in originals[:24]:
        mutated = mutate_statement(doc["text"])
        if mutated is None:
            continue
        new_text, rationale = mutated
        left_id = f"{doc['statement_id']}-std"
        right_id = f"{doc['statement_id']}-ent"
        docs.append(
            {
                "id": f"doc-{left_id}",
                "statement_id": left_id,
                "title": f"{doc['title']} (Standard plan)",
                "text": new_text.replace(f"[{doc['statement_id']}]", f"[{left_id}]", 1),
                "ci_smoke": False,
                "seeded": True,
                "near_miss": True,
            }
        )
        docs.append(
            {
                "id": f"doc-{right_id}",
                "statement_id": right_id,
                "title": f"{doc['title']} (Enterprise plan)",
                "text": doc["text"].replace(f"[{doc['statement_id']}]", f"[{right_id}]", 1),
                "ci_smoke": False,
                "seeded": True,
                "near_miss": True,
            }
        )
        near_miss_questions.append(
            {
                "id": f"pc-nm-{left_id}",
                "split": "pending",
                "task": "conflict_audit",
                "question": (
                    f"Do the Standard and Enterprise plan articles titled "
                    f"{doc['title']!r} contradict each other?"
                ),
                "gold": "No; they apply to different plans.",
                "gold_pairs": [],
                "non_pairs": [[left_id, right_id]],
                "scoring": "conflict_pairs",
                "gold_method": "plan_scoped_near_miss",
            }
        )
        if len(near_miss_questions) >= 20:
            break

    docs.extend(originals)
    docs.extend(seeded_docs)
    pair_questions: list[dict[str, Any]] = []
    scored_pairs = [row for row in gold_pairs if not row.get("ci_smoke")]
    for row in scored_pairs:
        pair_questions.append(
            {
                "id": f"pc-pair-{row['left']}-{row['right']}",
                "split": "pending",
                "task": "conflict_audit",
                "question": (
                    f"Do statements {row['left']} and {row['right']} contradict each other "
                    f"for the same audience and time? If yes, list the statement ids."
                ),
                "gold": f"{row['left']} vs {row['right']}",
                "gold_pairs": [[row["left"], row["right"]]],
                "must_include": [row["left"], row["right"]],
                "scoring": "conflict_pairs",
                "gold_method": "seeded" if row.get("seeded") else "natural_numeric_clash",
                "rationale": row.get("rationale"),
            }
        )
    questions = pair_questions + near_miss_questions
    meta = {
        "n_original_statements": len(originals),
        "n_seeded_twins": len(seeded_docs),
        "n_natural_pairs": natural,
        "n_gold_pairs": len(scored_pairs),
        "n_github_files": len(gh_files),
        "n_tts_files": len(tts_files),
    }
    return docs, gold_pairs, questions, meta


ACTION_RE = re.compile(
    r"\b(check|review|scale|restart|delete|increase|run|kubectl|describe|cordon|"
    r"drain|update|adjust|verify|inspect|roll|patch|apply|migrate|talk|see|add|"
    r"remove|set|use|follow|contact|wait|ensure|look|read|open|enable|disable|"
    r"fail|happen|occur)\b",
    re.I,
)
GOLD_STOP_END = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "of",
    "to",
    "in",
    "on",
    "for",
    "with",
    "from",
    "which",
    "that",
    "this",
    "can",
    "be",
    "is",
    "are",
}


def phrase_for_gold(text: str, *, min_words: int = 6, max_words: int = 16) -> str:
    words = _norm_space(text).split()
    if not words:
        return ""
    n = min(max_words, max(min_words, len(words)))
    n = min(n, len(words))
    while n >= min_words and words[n - 1].lower().strip(".,;:") in GOLD_STOP_END:
        n -= 1
    if n < min_words:
        n = min(min_words, len(words))
    return " ".join(words[:n])


def distinctive_action_span(source: str, other: str, *, min_words: int = 6, max_words: int = 16) -> str | None:
    other_l = other.lower()
    for sent in sentences(source) + [_norm_space(source)]:
        words = _norm_space(sent).split()
        upper = min(max_words, len(words))
        for n in range(upper, min_words - 1, -1):
            for i in range(0, len(words) - n + 1):
                chunk = words[i : i + n]
                if chunk[-1].lower().strip(".,;:") in GOLD_STOP_END:
                    continue
                span = " ".join(chunk)
                if not ACTION_RE.search(span):
                    continue
                if span.lower() in other_l:
                    continue
                return span
    words = _norm_space(source).split()
    for n in range(min(max_words, len(words)), min_words - 1, -1):
        for i in range(0, len(words) - n + 1):
            chunk = words[i : i + n]
            if chunk[-1].lower().strip(".,;:") in GOLD_STOP_END:
                continue
            span = " ".join(chunk)
            if span.lower() in other_l:
                continue
            return span
    return None


def repair_knowhow_questions(
    docs: list[dict[str, Any]], questions: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    by_id = {str(doc.get("id")): doc for doc in docs}
    repaired: list[dict[str, Any]] = []
    for row in questions:
        hops = [str(item) for item in (row.get("hops") or [])]
        if row.get("split") == "smoke" or len(hops) < 2:
            repaired.append(row)
            continue
        left = by_id.get(hops[0])
        right = by_id.get(hops[1])
        if not left or not right:
            continue
        left_text = str(left.get("text") or "")
        right_text = str(right.get("text") or "")
        mit = distinctive_action_span(left_text, right_text)
        diag = distinctive_action_span(right_text, left_text)
        if not mit or not diag:
            continue
        clone = dict(row)
        clone["must_include"] = [mit, diag]
        clone["gold"] = f"{mit} / {diag}"
        repaired.append(clone)
    return repaired


def parse_runbook(path: Path) -> dict[str, Any] | None:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    text = strip_markup(raw)
    if len(text) < 120:
        return None
    title = path.stem
    fm = FRONTMATTER_RE.search(raw)
    if fm:
        m = re.search(r"^title:\s*[\"']?(.+?)[\"']?\s*$", fm.group(0), re.M)
        if m:
            title = m.group(1).strip()
    sections: dict[str, str] = {}
    current = "body"
    buf: list[str] = []
    for line in raw.splitlines():
        heading = re.match(r"^##\s+(.+)", line)
        if heading:
            sections[current] = strip_markup("\n".join(buf))
            current = heading.group(1).strip().lower()
            buf = []
            continue
        buf.append(line)
    sections[current] = strip_markup("\n".join(buf))
    def first_sentence(key_opts: tuple[str, ...]) -> str:
        for key in key_opts:
            blob = sections.get(key) or ""
            sents = sentences(blob) or ([blob] if blob else [])
            for sent in sents:
                clean = _norm_space(sent)
                if len(clean) < 24 or clean.upper().startswith("TODO"):
                    continue
                if ACTION_RE.search(clean):
                    return phrase_for_gold(clean)
            for sent in sents:
                clean = _norm_space(sent)
                if len(clean) >= 24 and not clean.upper().startswith("TODO"):
                    return phrase_for_gold(clean)
        return ""

    mitigation = first_sentence(("mitigation", "mitigations", "response", "steps"))
    diagnosis = first_sentence(("diagnosis", "diagnostic", "meaning", "impact"))
    component = path.parent.name
    related = re.findall(r"\[([A-Za-z0-9_-]+)\]\([^)]*?\.md\)", raw)
    return {
        "id": f"rb-{component}-{path.stem}"[:80],
        "title": f"{component}: {title}",
        "text": text[:4000],
        "alert": path.stem,
        "component": component,
        "mitigation": mitigation,
        "diagnosis": diagnosis,
        "related": related,
        "source": "prometheus-operator/runbooks",
        "source_path": str(path),
        "license": "Apache-2.0",
        "ci_smoke": False,
    }


def build_knowhow(
    runbook_root: Path, existing_docs: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    paths = sorted((runbook_root / "content" / "runbooks").rglob("*.md"))
    runbooks: list[dict[str, Any]] = []
    for path in paths:
        if path.name.startswith("_"):
            continue
        parsed = parse_runbook(path)
        if parsed and parsed.get("mitigation") and parsed.get("diagnosis"):
            runbooks.append(parsed)
    by_component: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in runbooks:
        by_component[str(row["component"])].append(row)
    questions: list[dict[str, Any]] = []
    for component, group in sorted(by_component.items()):
        group = sorted(group, key=lambda row: str(row["alert"]))
        for idx, left in enumerate(group):
            neighbors = group[idx + 1 : idx + 6]
            if not neighbors and group:
                neighbors = [group[(idx + 1) % len(group)]]
            for jdx, right in enumerate(neighbors):
                if left["id"] == right["id"]:
                    continue
                mit = left.get("mitigation") or ""
                diag = right.get("diagnosis") or ""
                if len(mit) < 20 or len(diag) < 20:
                    continue
                mit_key = phrase_for_gold(mit)
                diag_key = phrase_for_gold(diag)
                if len(mit_key) < 12 or len(diag_key) < 12:
                    continue
                if mit_key.lower() in str(right.get("text") or "").lower():
                    continue
                if diag_key.lower() in str(left.get("text") or "").lower():
                    continue
                questions.append(
                    {
                        "id": f"ek-mh-{left['alert']}-{right['alert']}-{jdx}",
                        "split": "pending",
                        "task": "qa",
                        "question": (
                            f"A {component} incident fires both {left['alert']} and {right['alert']}. "
                            f"What is the first mitigation guidance in the {left['alert']} runbook, "
                            f"and what does the {right['alert']} runbook say first under diagnosis/meaning?"
                        ),
                        "gold": f"{mit_key} / {diag_key}",
                        "must_include": [mit_key, diag_key],
                        "scoring": "must_include",
                        "gold_method": "runbook_section_extraction_multi_hop",
                        "hops": [left["id"], right["id"]],
                    }
                )
            if left.get("related"):
                related_name = str(left["related"][0])
                match = next((row for row in runbooks if row["alert"] == related_name), None)
                if match and match["id"] != left["id"]:
                    mit_key = phrase_for_gold(str(left.get("mitigation") or ""))
                    diag_key = phrase_for_gold(str(match.get("diagnosis") or ""))
                    if (
                        len(mit_key) >= 12
                        and len(diag_key) >= 12
                        and mit_key.lower() not in str(match.get("text") or "").lower()
                        and diag_key.lower() not in str(left.get("text") or "").lower()
                    ):
                        questions.append(
                            {
                                "id": f"ek-rel-{left['alert']}-{match['alert']}",
                                "split": "pending",
                                "task": "qa",
                                "question": (
                                    f"The {left['alert']} runbook links to {match['alert']}. "
                                    f"What is {left['alert']}'s first mitigation step, and what is "
                                    f"{match['alert']}'s first diagnosis/meaning sentence?"
                                ),
                                "gold": f"{mit_key} / {diag_key}",
                                "must_include": [mit_key, diag_key],
                                "scoring": "must_include",
                                "gold_method": "runbook_link_multi_hop",
                                "hops": [left["id"], match["id"]],
                            }
                        )
    docs = list(existing_docs) + [
        {k: v for k, v in row.items() if k not in {"mitigation", "diagnosis", "related", "alert"}}
        | {
            "mitigation": row.get("mitigation"),
            "diagnosis": row.get("diagnosis"),
            "alert": row.get("alert"),
        }
        for row in runbooks
    ]
    questions = repair_knowhow_questions(docs, questions)
    meta = {
        "n_runbooks": len(runbooks),
        "n_candidate_questions": len(questions),
        "components": sorted(by_component),
    }
    return docs, questions, meta


def pick_spotcheck(rows: list[dict[str, Any]], n: int) -> list[dict[str, Any]]:
    if not rows:
        return []
    step = max(1, len(rows) // n)
    picked = [rows[idx] for idx in range(0, len(rows), step)][:n]
    return picked


def write_gold_md(
    path: Path,
    *,
    title: str,
    body: str,
    spot: list[dict[str, Any]],
    extra: str = "",
) -> None:
    lines = [f"# {title}", "", body.strip(), "", extra.strip(), "", "## Spot-check sample", ""]
    for idx, row in enumerate(spot, start=1):
        lines.append(f"### {idx}. `{row.get('id')}`")
        lines.append("")
        lines.append(f"- split: `{row.get('split')}`")
        lines.append(f"- gold method: `{row.get('gold_method')}`")
        if row.get("question"):
            lines.append(f"- question: {row['question']}")
        if row.get("gold"):
            lines.append(f"- gold: `{row['gold']}`")
        if row.get("must_include"):
            lines.append(f"- must_include: `{row['must_include']}`")
        if row.get("gold_pairs"):
            lines.append(f"- gold_pairs: `{row['gold_pairs']}`")
        if row.get("review"):
            lines.append(f"- review: {row['review']}")
        lines.append("")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--github-docs", type=Path, default=DEFAULT_CLONES["github_docs"])
    parser.add_argument("--tts-handbook", type=Path, default=DEFAULT_CLONES["tts_handbook"])
    parser.add_argument("--runbooks", type=Path, default=DEFAULT_CLONES["po_runbooks"])
    parser.add_argument("--cache", type=Path, default=CACHE / "far")
    parser.add_argument("--skip-far-fetch", action="store_true")
    args = parser.parse_args(argv)

    vr_dir = DATASETS / "versioned_rules"
    pc_dir = DATASETS / "policy_conflicts"
    ek_dir = DATASETS / "expert_knowhow"

    smoke_vr_docs = load_smoke_docs(vr_dir / "corpus.jsonl")
    smoke_vr_qs = load_smoke_questions(vr_dir / "questions.jsonl")
    smoke_pc_docs = load_smoke_docs(pc_dir / "corpus.jsonl")
    smoke_pc_qs = load_smoke_questions(pc_dir / "questions.jsonl")
    smoke_pc_pairs = load_smoke_pairs(pc_dir / "gold_pairs.jsonl")
    smoke_ek_docs = load_smoke_docs(ek_dir / "corpus.jsonl")
    smoke_ek_qs = load_smoke_questions(ek_dir / "questions.jsonl")

    print("Building FAR/eCFR versioned rules…")
    far_docs, far_qs, far_meta = build_far(args.cache, skip_fetch=args.skip_far_fetch)
    if args.skip_far_fetch and not far_docs:
        print(f"No cached FAR XML under {args.cache}; leaving versioned_rules unchanged")
        vr_qs = _read_jsonl(vr_dir / "questions.jsonl")
        far_meta = {"skipped_fetch": True, "n_docs": 0}
    else:
        far_qs = split_assign(far_qs, n_dev=40, n_test=180)
        vr_docs = smoke_vr_docs + far_docs
        vr_qs = smoke_vr_qs + far_qs
        _write_jsonl(vr_dir / "corpus.jsonl", vr_docs)
        _write_jsonl(vr_dir / "questions.jsonl", vr_qs)
        (vr_dir / "far_fetch.json").write_text(json.dumps(far_meta, indent=2) + "\n", encoding="utf-8")

    print("Building policy contradiction corpus…")
    pc_docs, pc_pairs, pc_qs, pc_meta = build_policy(
        args.github_docs, args.tts_handbook, smoke_pc_docs, smoke_pc_pairs
    )
    # Keep smoke questions, then 30 dev + 150 test from generated.
    generated = split_fill_test(pc_qs, n_dev=30, min_test=150, max_test=180)
    pc_questions = smoke_pc_qs + generated
    _write_jsonl(pc_dir / "corpus.jsonl", pc_docs)
    _write_jsonl(pc_dir / "gold_pairs.jsonl", pc_pairs)
    _write_jsonl(pc_dir / "questions.jsonl", pc_questions)
    (pc_dir / "build_meta.json").write_text(json.dumps(pc_meta, indent=2) + "\n", encoding="utf-8")

    print("Building expert know-how runbooks…")
    ek_docs, ek_qs, ek_meta = build_knowhow(args.runbooks, smoke_ek_docs)
    ek_assigned = split_fill_test(ek_qs, n_dev=30, min_test=150, max_test=180)
    _write_jsonl(ek_dir / "corpus.jsonl", ek_docs)
    _write_jsonl(ek_dir / "questions.jsonl", smoke_ek_qs + ek_assigned)
    (ek_dir / "build_meta.json").write_text(json.dumps(ek_meta, indent=2) + "\n", encoding="utf-8")

    def counts(name: str, qpath: Path, cpath: Path, pairs: Path | None = None) -> None:
        qs = _read_jsonl(qpath)
        by = defaultdict(int)
        for row in qs:
            by[str(row.get("split"))] += 1
        extra = f" pairs={len(_read_jsonl(pairs))}" if pairs and pairs.exists() else ""
        print(f"{name}: docs={len(_read_jsonl(cpath))} splits={dict(by)}{extra}")

    counts("versioned_rules", vr_dir / "questions.jsonl", vr_dir / "corpus.jsonl")
    counts("policy_conflicts", pc_dir / "questions.jsonl", pc_dir / "corpus.jsonl", pc_dir / "gold_pairs.jsonl")
    counts("expert_knowhow", ek_dir / "questions.jsonl", ek_dir / "corpus.jsonl")

    vr_spot = pick_spotcheck([row for row in vr_qs if row.get("split") == "test"], 12)
    pc_spot = pick_spotcheck([row for row in pc_questions if row.get("split") == "test"], 12)
    ek_spot = pick_spotcheck([row for row in ek_assigned if row.get("split") == "test"], 12)
    _write_jsonl(vr_dir / "spotcheck.jsonl", vr_spot)
    _write_jsonl(pc_dir / "spotcheck.jsonl", pc_spot)
    _write_jsonl(ek_dir / "spotcheck.jsonl", ek_spot)

    write_gold_md(
        vr_dir / "GOLD.md",
        title="Versioned rules gold production",
        body="""
Gold for the scored FAR split is derived **programmatically from eCFR Title 48 Chapter 1 XML**.

1. Fetch part-level XML from `https://www.ecfr.gov/api/versioner/v1/full/{date}/title-48.xml?chapter=1&part={part}` for snapshots listed in `far_fetch.json`.
2. Parse `DIV*` nodes with `TYPE=SECTION`. Consecutive snapshots with identical normalized text are merged into one validity interval (`valid_from` / `valid_to`).
3. **As-of gold:** a numeric/unit span that appears in the in-force version and in **no other** retained version of that clause.
4. **Changelog gold:** spans (or sentences) present in version N but not N+1, and vice versa.
5. FAMR fixtures remain `ci_smoke` only and are **not** part of the scored test split.

U.S. FAR/eCFR text is a U.S. government work (17 U.S.C. § 105).
""",
        extra=f"Builder meta: `{json.dumps({k: far_meta[k] for k in far_meta if k != 'fetch_log'})}`",
        spot=vr_spot,
    )
    write_gold_md(
        pc_dir / "GOLD.md",
        title="Policy conflict gold production",
        body="""
The scored corpus is built from **GitHub Docs** (CC-BY-4.0 help center) and the **GSA-TTS handbook** (U.S. government work / CC0 policy).

1. Extract short paragraphs that contain a number+unit or required/optional polarity.
2. **Seeded pairs (majority):** duplicate the paragraph with a single documented mutation (number change or required↔optional) stored as a `stale copy`. Gold pair = original statement_id vs `*-seed`.
3. **Natural pairs:** two unmodified paragraphs from the same source whose titles share a *specific* topic token (generic words like “about” do not count) and whose numeric facts are disjoint **but in the same unit family** (duration vs duration, money vs money). Pairs that only share a generic title word, or that mix unrelated units (e.g. 2FA days vs refund days, $1 hold vs 10 business days), are not gold.
4. **Near-misses:** Standard vs Enterprise plan titles with different numbers — listed on questions as `non_pairs`, **not** in `gold_pairs.jsonl`.
5. Original Northwind fixtures remain `ci_smoke`.

Every committed gold pair was produced by this script; a sample is reviewed in this file.
""",
        extra=f"Builder meta: `{json.dumps(pc_meta)}`",
        spot=pc_spot,
    )
    write_gold_md(
        ek_dir / "GOLD.md",
        title="Expert know-how gold production",
        body="""
Scored items are multi-hop questions over **prometheus-operator/runbooks** (Apache-2.0).

1. Parse each runbook's `## Mitigation` (or Response/Steps) and `## Diagnosis`/`## Meaning` first sentence.
2. Pair two runbooks in the same component (or a runbook and a markdown-linked peer).
3. Gold `must_include` spans are action-bearing phrases unique to each hop document (not truncated mid-function-word). Answering requires both documents.
4. Original Northwind SRE/lab fixtures remain `ci_smoke`.
""",
        extra=f"Builder meta: `{json.dumps(ek_meta)}`",
        spot=ek_spot,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
