from __future__ import annotations

import pytest

from scripts.release_version import ReleaseVersion, parse_release_tag


@pytest.mark.parametrize(
    ("tag", "expected"),
    [
        ("v2.19.0", ReleaseVersion("2.19.0", "2.19.0", "stable")),
        ("v2.19.0-rc.1", ReleaseVersion("2.19.0-rc.1", "2.19.0rc1", "rc")),
        ("v2.19.0-dev", ReleaseVersion("2.19.0-dev", "2.19.0.dev0", "dev")),
        ("v2.19.0-dev.3", ReleaseVersion("2.19.0-dev.3", "2.19.0.dev3", "dev")),
    ],
)
def test_parse_release_tag(tag: str, expected: ReleaseVersion):
    assert parse_release_tag(tag) == expected


@pytest.mark.parametrize(
    "tag",
    [
        "2.19.0",
        "v2.19",
        "v2.19.0-rc",
        "v2.19.0-dev.foo",
        "v2.19.0-beta.1",
    ],
)
def test_parse_release_tag_rejects_unsupported_formats(tag: str):
    with pytest.raises(ValueError, match="Unsupported release tag"):
        parse_release_tag(tag)
