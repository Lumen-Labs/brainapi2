from __future__ import annotations

import importlib
import logging
import shutil
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from src.core.plugins.catalog import PluginManifestCatalog
from src.core.plugins.entrypoints import PluginEntryPointResolver
from src.core.plugins.manifest import PluginManifest

if TYPE_CHECKING:
    from src.core.plugins.context import PluginContext

logger = logging.getLogger("brainapi.plugins")


class PluginLoader:
    def __init__(
        self,
        plugins_dir: Path,
        context: "PluginContext",
        manifest_catalog: Optional[PluginManifestCatalog] = None,
        entry_point_resolver: Optional[PluginEntryPointResolver] = None,
    ):
        self.plugins_dir = plugins_dir
        self.context = context
        self._loaded: dict[str, PluginManifest] = {}
        self._manifest_catalog = manifest_catalog or PluginManifestCatalog(
            plugins_dir=plugins_dir,
            logger_override=logger,
        )
        self._entry_point_resolver = entry_point_resolver or PluginEntryPointResolver()

    def discover(self) -> list[PluginManifest]:
        if not self.plugins_dir.exists():
            logger.info("Plugins directory '%s' does not exist, skipping discovery", self.plugins_dir)
            return []
        manifests = self._manifest_catalog.discover(validate=True)
        manifests.sort(key=lambda m: m.priority)
        return manifests

    def install_dependencies(self, manifest: PluginManifest) -> None:
        if not manifest.pip_dependencies:
            return
        logger.info("Installing pip dependencies for plugin '%s': %s", manifest.name, manifest.pip_dependencies)
        cmd = self._build_install_cmd(manifest.pip_dependencies)
        try:
            subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as exc:
            logger.error(
                "Failed to install dependencies for plugin '%s': %s",
                manifest.name,
                exc.stderr.decode() if exc.stderr else str(exc),
            )
            raise

    @staticmethod
    def _build_install_cmd(packages: list[str]) -> list[str]:
        if shutil.which("uv"):
            return ["uv", "pip", "install", "--quiet", *packages]
        return [sys.executable, "-m", "pip", "install", "--quiet", *packages]

    def load(self, manifest: PluginManifest) -> bool:
        if manifest.name in self._loaded:
            logger.debug("Plugin '%s' already loaded, skipping", manifest.name)
            return True

        plugin_dir = manifest.plugin_dir
        if plugin_dir is None:
            logger.error("Plugin '%s' has no plugin_dir set", manifest.name)
            return False

        try:
            self.install_dependencies(manifest)
        except Exception:
            return False

        plugin_dir_str = str(plugin_dir)
        if plugin_dir_str not in sys.path:
            sys.path.insert(0, plugin_dir_str)

        module_name = self._entry_point_resolver.resolve(
            entry_point=manifest.entry_point,
            plugin_dir=plugin_dir,
        )

        try:
            module = importlib.import_module(module_name)
        except Exception as exc:
            logger.error("Failed to import plugin '%s' (module: %s): %s", manifest.name, module_name, exc)
            if plugin_dir_str in sys.path:
                sys.path.remove(plugin_dir_str)
            return False

        register_fn = getattr(module, "register", None)
        if register_fn is None or not callable(register_fn):
            logger.error("Plugin '%s' has no callable 'register' function in '%s'", manifest.name, module_name)
            if plugin_dir_str in sys.path:
                sys.path.remove(plugin_dir_str)
            return False

        try:
            register_fn(self.context)
            self._loaded[manifest.name] = manifest
            logger.info("Plugin '%s' v%s loaded successfully", manifest.name, manifest.version)
            return True
        except Exception as exc:
            logger.error("Plugin '%s' register() raised an exception: %s", manifest.name, exc, exc_info=True)
            if plugin_dir_str in sys.path:
                sys.path.remove(plugin_dir_str)
            return False

    def load_all(self) -> dict[str, bool]:
        manifests = self.discover()
        results: dict[str, bool] = {}
        for manifest in manifests:
            try:
                results[manifest.name] = self.load(manifest)
            except Exception as exc:
                logger.error("Unexpected error loading plugin '%s': %s", manifest.name, exc, exc_info=True)
                results[manifest.name] = False
        loaded_count = sum(1 for v in results.values() if v)
        total = len(results)
        if total > 0:
            logger.info("Loaded %d/%d plugins", loaded_count, total)
        return results

    @property
    def loaded_plugins(self) -> dict[str, PluginManifest]:
        return dict(self._loaded)
