from __future__ import annotations

from pathlib import Path


class PluginEntryPointResolver:
    def resolve(self, entry_point: str, plugin_dir: Path) -> str:
        normalized = entry_point.strip().replace("\\", "/")
        if normalized.endswith(".py"):
            normalized = normalized[:-3]
        normalized = normalized.lstrip("./").strip("/")
        module_name = normalized.replace("/", ".")
        if module_name.endswith(".__init__"):
            module_name = module_name[: -len(".__init__")]
        if module_name == "__init__" or module_name == "":
            return plugin_dir.name
        return module_name
