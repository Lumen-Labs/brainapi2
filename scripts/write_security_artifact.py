#!/usr/bin/env python3
"""Summarize an exact image's Trivy scan, digest, and SBOM for release gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trivy", type=Path, required=True)
    parser.add_argument("--sbom", type=Path, required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    scan = json.loads(args.trivy.read_text(encoding="utf-8"))
    counts = {"HIGH": 0, "CRITICAL": 0}
    for result in scan.get("Results") or []:
        for vulnerability in result.get("Vulnerabilities") or []:
            severity = str(vulnerability.get("Severity", "")).upper()
            if severity in counts:
                counts[severity] += 1
    digest = subprocess.run(
        ["docker", "image", "inspect", args.image, "--format", "{{.Id}}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    archive_sha256 = ""
    if args.archive:
        hasher = hashlib.sha256()
        with args.archive.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                hasher.update(chunk)
        archive_sha256 = hasher.hexdigest()
    result = {
        "high": counts["HIGH"],
        "critical": counts["CRITICAL"],
        "image_digest": digest,
        "archive_sha256": archive_sha256,
        "sbom": args.sbom.name,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
