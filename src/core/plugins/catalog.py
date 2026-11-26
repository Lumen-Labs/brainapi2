from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from src.core.plugins.manifest import MANIFEST_FILENAME, PluginManifest, parse_manifest

logger = logging.getLogger("brainapi.plugins.catalog")


class PluginManifestCatalog:
    def __init__(
        self,
        plugins_dir: Path,
        logger_override: Optional[logging.Logger] = None,
        extra_dirs: Optional[list[Path]] = None,
    ):
        self.plugins_dir = plugins_dir
        self.extra_dirs = list(extra_dirs or [])
        self._logger = logger_override or logger

    def discover(self, validate: bool = False) -> list[PluginManifest]:
        by_name: dict[str, PluginManifest] = {}
        for directory in [self.plugins_dir, *self.extra_dirs]:
            for manifest in self._discover_one(directory, validate=validate):
                by_name[manifest.name] = manifest
        return list(by_name.values())

    def _discover_one(self, plugins_dir: Path, validate: bool) -> list[PluginManifest]:
        if not plugins_dir.exists():
            return []

        manifests: list[PluginManifest] = []
        for child in sorted(plugins_dir.iterdir()):
            if not child.is_dir():
                continue
            manifest_path = child / MANIFEST_FILENAME
            if not manifest_path.exists():
                continue
            try:
                manifest = parse_manifest(manifest_path)
            except Exception as exc:
                self._logger.warning("Failed to parse manifest in '%s': %s", child.name, exc)
                continue

            if validate:
                errors = manifest.validate()
                if errors:
                    self._logger.warning(
                        "Plugin '%s' manifest validation failed: %s",
                        child.name,
                        "; ".join(errors),
                    )
                    continue

            manifests.append(manifest)

        return manifests
