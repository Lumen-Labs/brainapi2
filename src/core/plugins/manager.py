from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Optional

from src.core.plugins.manifest import MANIFEST_FILENAME, PluginManifest, parse_manifest
from src.core.plugins.registry import PluginMeta, PluginRegistryClient

logger = logging.getLogger("brainapi.plugins.manager")


class PluginManager:
    def __init__(
        self,
        plugins_dir: Path,
        registry_url: Optional[str] = None,
        publisher_id: Optional[str] = None,
        publisher_api_key: Optional[str] = None,
    ):
        self.plugins_dir = plugins_dir
        self.registry_url = registry_url
        self._registry: Optional[PluginRegistryClient] = None
        if registry_url:
            self._registry = PluginRegistryClient(
                registry_url=registry_url,
                publisher_id=publisher_id,
                api_key=publisher_api_key,
            )

    def _ensure_registry(self) -> PluginRegistryClient:
        if self._registry is None:
            raise RuntimeError(
                "No plugin registry configured. "
                "Set PLUGIN_REGISTRY_URL environment variable or pass registry_url."
            )
        return self._registry

    def install(self, name: str, version: str = "latest") -> PluginManifest:
        existing_dir = self.plugins_dir / name
        existing_manifest = existing_dir / MANIFEST_FILENAME
        if existing_manifest.exists():
            try:
                current = parse_manifest(existing_manifest)
                if version == "latest" or current.version == version:
                    logger.info("Plugin '%s' v%s already installed, skipping", name, current.version)
                    return current
            except Exception:
                pass

        registry = self._ensure_registry()
        plugin_dir = registry.download(name, version=version, target_dir=self.plugins_dir)

        manifest_path = plugin_dir / MANIFEST_FILENAME
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Downloaded plugin '{name}' does not contain a {MANIFEST_FILENAME}"
            )

        manifest = parse_manifest(manifest_path)
        errors = manifest.validate()
        if errors:
            shutil.rmtree(plugin_dir, ignore_errors=True)
            raise ValueError(
                f"Plugin '{name}' manifest validation failed: {'; '.join(errors)}"
            )

        logger.info("Installed plugin '%s' v%s to '%s'", manifest.name, manifest.version, plugin_dir)
        return manifest

    def uninstall(self, name: str) -> bool:
        plugin_dir = self.plugins_dir / name
        if not plugin_dir.exists():
            logger.warning("Plugin '%s' not found in '%s'", name, self.plugins_dir)
            return False

        shutil.rmtree(plugin_dir)
        logger.info("Uninstalled plugin '%s'", name)
        return True

    def list_installed(self) -> list[PluginManifest]:
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
                manifests.append(parse_manifest(manifest_path))
            except Exception as exc:
                logger.warning("Skipping '%s': %s", child.name, exc)

        return manifests

    def list_available(self) -> list[PluginMeta]:
        registry = self._ensure_registry()
        return registry.list_plugins()

    def get_info(self, name: str) -> PluginMeta:
        registry = self._ensure_registry()
        return registry.get_plugin_info(name)

    def update(self, name: str) -> PluginManifest:
        return self.install(name, version="latest")

    def publish(self, archive_path: Path, name: Optional[str] = None, force: bool = False) -> dict:
        registry = self._ensure_registry()
        return registry.publish(archive_path, name=name, force=force)

    def delete_remote(self, name: str, version: Optional[str] = None) -> dict:
        registry = self._ensure_registry()
        return registry.delete(name, version=version)

    def register_publisher(self, publisher_id: Optional[str] = None) -> dict:
        if not self.registry_url:
            raise RuntimeError(
                "No plugin registry configured. "
                "Set PLUGIN_REGISTRY_URL environment variable or pass registry_url."
            )
        client = PluginRegistryClient(registry_url=self.registry_url)
        return client.register_publisher(publisher_id=publisher_id)
