#!/usr/bin/env python3
"""Validate the recorded product, restore, security, and latency release gates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COUNT_KEYS = ("brains", "nodes", "edges", "chunks", "vectors", "observations")
REQUIRED_FLOWS = {
    "authentication",
    "brain_creation",
    "plain_ingestion",
    "structured_ingestion",
    "task_completion",
    "context_retrieval",
    "search",
    "console",
    "mcp",
}


def _read(path: Path) -> dict:
    if not path.is_file():
        raise RuntimeError(f"Missing release artifact: {path.name}")
    return json.loads(path.read_text(encoding="utf-8"))


def _check_public_ledgers() -> None:
    reports = _read(ROOT / "benchmarks" / "REPORTS.json")["benchmarks"]
    longmem = reports["longmemeval"].get("leaderboard", [])
    if not longmem:
        raise RuntimeError("A representative LongMemEval result has not been published")
    search = reports["search"]
    representative = search.get("representative_run_id")
    row = next(
        (item for item in search.get("leaderboard", []) if item.get("run_id") == representative),
        None,
    )
    if not row or "wands" not in str(row.get("dataset", "")).lower():
        raise RuntimeError("The representative Search ledger entry must be WANDS")


def _check_smoke(data: dict) -> None:
    for profile in ("light", "heavy"):
        result = data.get(profile, {})
        if result.get("unexpected_5xx") != 0:
            raise RuntimeError(f"{profile} smoke recorded unexpected 5xx responses")
        passed = {name for name, ok in result.get("flows", {}).items() if ok}
        missing = REQUIRED_FLOWS - passed
        if missing:
            raise RuntimeError(f"{profile} smoke is missing flows: {sorted(missing)}")


def _check_latency(data: dict) -> None:
    if "light" in data or "heavy" in data:
        for profile in ("light", "heavy"):
            if profile not in data:
                raise RuntimeError(f"Latency results are missing the {profile} profile")
            _check_latency_result(profile, data[profile])
        return
    _check_latency_result("release", data)


def _check_latency_result(profile: str, data: dict) -> None:
    context = data.get("context", {})
    search = data.get("search", {})
    if float(context.get("p50_ms", float("inf"))) >= 1000:
        raise RuntimeError(f"{profile} /retrieve/context p50 must be below 1000 ms")
    if int(context.get("online_llm_retrieval_loops", -1)) != 0:
        raise RuntimeError(f"{profile} context retrieval must not run an online LLM loop")
    if float(search.get("p50_ms", float("inf"))) >= 200:
        raise RuntimeError(f"{profile} default search p50 must be below 200 ms")
    if search.get("excludes_embed_query") is not True:
        raise RuntimeError(f"{profile} search latency must exclude embed.query")
    for surface, result in (("context", context), ("search", search)):
        if "p95_ms" not in result or "p99_ms" not in result:
            raise RuntimeError(
                f"{profile} {surface} must record non-blocking p95 and p99"
            )


def _check_restore(profile: str, data: dict) -> None:
    before = data.get("before", {})
    after = data.get("after", {})
    for key in COUNT_KEYS:
        if before.get(key) != after.get(key):
            raise RuntimeError(f"{profile} restore count mismatch: {key}")
    checks = data.get("retrieval_spot_checks", [])
    if len(checks) < 10 or not all(item.get("match") for item in checks[:10]):
        raise RuntimeError(f"{profile} restore needs ten matching retrieval spot checks")


def _check_security(data: dict, artifact_dir: Path) -> None:
    if data.get("high") != 0 or data.get("critical") != 0:
        raise RuntimeError("The exact image has high or critical advisories")
    digest = str(data.get("image_digest", ""))
    if not digest.startswith("sha256:"):
        raise RuntimeError("The exact image digest was not recorded")
    archive_sha256 = str(data.get("archive_sha256", ""))
    if len(archive_sha256) != 64:
        raise RuntimeError("The exact image archive hash was not recorded")
    sbom = artifact_dir / str(data.get("sbom", ""))
    if not sbom.is_file():
        raise RuntimeError("The exact image SBOM is missing")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact_dir", type=Path)
    args = parser.parse_args()
    artifact_dir = args.artifact_dir.resolve()
    try:
        _check_public_ledgers()
        _check_smoke(_read(artifact_dir / "smoke.json"))
        _check_latency(_read(artifact_dir / "latency.json"))
        _check_restore("light", _read(artifact_dir / "restore-light.json"))
        _check_restore("heavy", _read(artifact_dir / "restore-heavy.json"))
        _check_security(_read(artifact_dir / "security.json"), artifact_dir)
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        print(f"release gate failed: {exc}", file=sys.stderr)
        return 1
    print("All production release gates passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
