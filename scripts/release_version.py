"""Parse container release tags and normalize them to Python project versions."""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass


_STABLE_TAG = re.compile(r"^v(?P<base>\d+\.\d+\.\d+)$")
_RC_TAG = re.compile(r"^v(?P<base>\d+\.\d+\.\d+)-rc\.(?P<number>\d+)$")
_DEV_TAG = re.compile(
    r"^v(?P<base>\d+\.\d+\.\d+)-dev(?:\.(?P<number>\d+))?$"
)


@dataclass(frozen=True)
class ReleaseVersion:
    tag_version: str
    project_version: str
    channel: str


PRODUCT_CLAIM_CHANNELS = frozenset({"stable", "rc"})


def parse_release_tag(tag: str) -> ReleaseVersion:
    """Return the container tag version and its PEP 440 project equivalent."""
    if match := _STABLE_TAG.fullmatch(tag):
        base = match.group("base")
        return ReleaseVersion(tag_version=base, project_version=base, channel="stable")

    if match := _RC_TAG.fullmatch(tag):
        base = match.group("base")
        number = match.group("number")
        return ReleaseVersion(
            tag_version=f"{base}-rc.{number}",
            project_version=f"{base}rc{number}",
            channel="rc",
        )

    if match := _DEV_TAG.fullmatch(tag):
        base = match.group("base")
        number = match.group("number") or "0"
        tag_version = f"{base}-dev"
        if match.group("number") is not None:
            tag_version = f"{tag_version}.{number}"
        return ReleaseVersion(
            tag_version=tag_version,
            project_version=f"{base}.dev{number}",
            channel="dev",
        )

    raise ValueError(
        f"Unsupported release tag: {tag}. Expected vX.Y.Z, vX.Y.Z-rc.N, "
        "vX.Y.Z-dev, or vX.Y.Z-dev.N"
    )


def requires_product_claim_gates(tag: str | None) -> bool:
    if not tag:
        return True
    return parse_release_tag(tag).channel in PRODUCT_CLAIM_CHANNELS


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(f"Usage: {argv[0]} <release-tag>", file=sys.stderr)
        return 2
    try:
        release = parse_release_tag(argv[1])
    except ValueError as error:
        print(error, file=sys.stderr)
        return 2
    print(f"version={release.tag_version}")
    print(f"project_version={release.project_version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
