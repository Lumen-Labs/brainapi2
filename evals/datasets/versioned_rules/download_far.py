#!/usr/bin/env python3
"""Fetch public-domain FAR HTML from acquisition.gov (U.S. government work).

This script does not invent gold as-of labels. It writes a manifest with URL,
fetch time, and HTTP status so a later human pass can align historical versions.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_CLAUSES = [
    "https://www.acquisition.gov/far/52.212-4",
    "https://www.acquisition.gov/far/52.203-13",
    "https://www.acquisition.gov/far/52.217-9",
    "https://www.acquisition.gov/far/part-12",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "downloads",
    )
    parser.add_argument("--url", action="append", dest="urls")
    parser.add_argument("--sleep", type=float, default=1.0)
    args = parser.parse_args()
    urls = args.urls or list(DEFAULT_CLAUSES)
    args.out.mkdir(parents=True, exist_ok=True)

    try:
        import httpx
    except ImportError as exc:
        raise SystemExit("httpx is required. pip install -r evals/requirements.txt") from exc

    manifest = []
    with httpx.Client(timeout=60.0, follow_redirects=True) as client:
        for url in urls:
            started = time.perf_counter()
            response = client.get(url, headers={"User-Agent": "brainapi-evals/0.1"})
            elapsed = time.perf_counter() - started
            slug = url.rstrip("/").split("/")[-1].replace(".", "_")
            body_path = args.out / f"{slug}.html"
            body_path.write_bytes(response.content)
            row = {
                "url": url,
                "path": str(body_path),
                "status_code": response.status_code,
                "fetched_at": datetime.now(timezone.utc).isoformat(),
                "bytes": len(response.content),
                "elapsed_s": elapsed,
                "license": "U.S. government work, 17 U.S.C. § 105 public domain in the U.S.",
                "gold_labeled": False,
            }
            manifest.append(row)
            print(f"{response.status_code} {url} -> {body_path}")
            time.sleep(max(0.0, args.sleep))
    man_path = args.out / "manifest.json"
    man_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {man_path}. Gold labels are NOT inferred; fixture questions remain the scored set.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
