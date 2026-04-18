from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from src.core.plugins.manifest import MANIFEST_FILENAME, PluginManifest, parse_manifest

logger = logging.getLogger("brainapi.plugins.catalog")


class PluginManifestCatalog:
    def __init__(self, plugins_dir: Path, logger_override: Optional[logging.Logger] = None):
        self.plugins_dir = plugins_dir
        self._logger = logger_override or logger

    def discover(self, validate: bool = False) -> list[PluginManifest]:
        if not self.plugins_dir.exists():
            return []

        manifests: list[PluginManifest] = []
        for child in sorted(self.plugins_dir.iterdir()):
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
