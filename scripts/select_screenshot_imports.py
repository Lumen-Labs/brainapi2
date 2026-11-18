#!/usr/bin/env python3
"""Select screenshot-fixture imports on a named brain for a manual ops dry-run.

This is not part of the product runtime. Pass a job list (task rows) and a brain
id; nothing here deletes data or holds credentials.
"""

from __future__ import annotations

from typing import Iterable, Optional


SCREENSHOT_IMPORTS = (
    {"label": "Slack message", "chars": 26},
    {"contains": "6612-WALRUS", "chars": 40},
    {"contains": "joined the channel", "chars": 37},
    {"contains": "2204-MINK", "chars": 34},
    {"label": "Connect your tools", "chars": 270},
    {"label": "Connect your tools", "chars": 1273},
    {"label": "Sync probe cedar", "chars": 66},
)

TARGET_BRAIN = "6aeae5d478a5d35c3d03a00e"


def matches_screenshot_import(label: str, chars: Optional[int]) -> bool:
    text = (label or "").strip()
    for spec in SCREENSHOT_IMPORTS:
        if chars != spec["chars"]:
            continue
        exact = spec.get("label")
        if exact is not None and text == exact:
            return True
        needle = spec.get("contains")
        if needle and needle in text:
            return True
    return False


def select_screenshot_jobs(
    jobs: Iterable[dict],
    *,
    brain_id: str = TARGET_BRAIN,
) -> list[dict]:
    chosen: list[dict] = []
    for job in jobs:
        if str(job.get("brain_id") or job.get("brainRef") or "") != brain_id:
            continue
        chars = job.get("chars")
        if chars is None:
            size = job.get("size") or {}
            chars = size.get("chars")
        if matches_screenshot_import(
            str(job.get("label") or ""),
            chars if chars is None else int(chars),
        ):
            chosen.append(job)
    return chosen


def main() -> None:
    import argparse
    import json
    import sys

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--brain",
        default=TARGET_BRAIN,
        help="Brain id to match (default: the screenshot-fixture brain)",
    )
    parser.add_argument(
        "jobs_json",
        nargs="?",
        help="Path to a JSON list of task rows, or stdin when omitted",
    )
    args = parser.parse_args()
    if args.jobs_json:
        with open(args.jobs_json, encoding="utf-8") as handle:
            raw = handle.read()
    else:
        raw = sys.stdin.read()
    jobs = json.loads(raw)
    if not isinstance(jobs, list):
        raise SystemExit("jobs JSON must be a list")
    selected = select_screenshot_jobs(jobs, brain_id=args.brain)
    json.dump(selected, sys.stdout, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
