from __future__ import annotations

import importlib.util
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
        added_plugin_path = False
        if plugin_dir_str not in sys.path:
            sys.path.insert(0, plugin_dir_str)
            added_plugin_path = True
        self._clear_local_plugin_namespaces()

        module_name = self._entry_point_resolver.resolve(
            entry_point=manifest.entry_point,
            plugin_dir=plugin_dir,
        )
        entrypoint_path = self._entry_point_resolver.resolve_path(
            entry_point=manifest.entry_point,
            plugin_dir=plugin_dir,
        )
        unique_module_name = self._module_name_for_manifest(manifest, module_name)

        try:
            module = self._import_entrypoint(
                unique_module_name=unique_module_name,
                entrypoint_path=entrypoint_path,
            )
        except Exception as exc:
            logger.error("Failed to import plugin '%s' (module: %s): %s", manifest.name, module_name, exc)
            if added_plugin_path and plugin_dir_str in sys.path:
                sys.path.remove(plugin_dir_str)
            return False

        register_fn = getattr(module, "register", None)
        if register_fn is None or not callable(register_fn):
            logger.error("Plugin '%s' has no callable 'register' function in '%s'", manifest.name, module_name)
            if added_plugin_path and plugin_dir_str in sys.path:
                sys.path.remove(plugin_dir_str)
            return False

        try:
            register_fn(self.context)
            self._loaded[manifest.name] = manifest
            logger.info("Plugin '%s' v%s loaded successfully", manifest.name, manifest.version)
            if added_plugin_path and plugin_dir_str in sys.path:
                sys.path.remove(plugin_dir_str)
            return True
        except Exception as exc:
            logger.error("Plugin '%s' register() raised an exception: %s", manifest.name, exc, exc_info=True)
            if added_plugin_path and plugin_dir_str in sys.path:
                sys.path.remove(plugin_dir_str)
            return False

    def _clear_local_plugin_namespaces(self) -> None:
        namespace_roots = (
            "routes",
            "controllers",
            "adapters",
            "constants",
            "agents",
            "workers",
            "prompts",
        )
        for module_name in list(sys.modules.keys()):
            if any(
                module_name == namespace_root
                or module_name.startswith(f"{namespace_root}.")
                for namespace_root in namespace_roots
            ):
                sys.modules.pop(module_name, None)

    @staticmethod
    def _safe_module_token(value: str) -> str:
        return "".join(ch if ch.isalnum() else "_" for ch in value)

    def _module_name_for_manifest(self, manifest: PluginManifest, module_name: str) -> str:
        plugin_token = self._safe_module_token(manifest.name)
        module_token = self._safe_module_token(module_name)
        return f"brainapi_plugin_{plugin_token}__{module_token}"

    def _import_entrypoint(
        self,
        unique_module_name: str,
        entrypoint_path: Path,
    ):
        if not entrypoint_path.exists():
            raise ModuleNotFoundError(f"Entrypoint file not found: {entrypoint_path}")
        spec = importlib.util.spec_from_file_location(unique_module_name, entrypoint_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to create module spec for '{entrypoint_path}'")
        module = importlib.util.module_from_spec(spec)
        sys.modules[unique_module_name] = module
        spec.loader.exec_module(module)
        return module

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
