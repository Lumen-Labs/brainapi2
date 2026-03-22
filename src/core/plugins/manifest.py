from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


MANIFEST_FILENAME = "plugin.yaml"

REQUIRED_FIELDS = ("name", "version", "entry_point")


@dataclass
class PluginManifest:
    name: str
    version: str
    entry_point: str
    description: str = ""
    author: str = ""
    brainapi_version: str = ""
    priority: int = 100
    pip_dependencies: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    plugin_dir: Optional[Path] = None

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not self.name or not self.name.strip():
            errors.append("'name' is required and cannot be empty")
        if not self.version or not self.version.strip():
            errors.append("'version' is required and cannot be empty")
        if not self.entry_point or not self.entry_point.strip():
            errors.append("'entry_point' is required and cannot be empty")
        if self.plugin_dir and not self.plugin_dir.is_dir():
            errors.append(f"plugin_dir '{self.plugin_dir}' is not a valid directory")
        return errors


def parse_manifest(manifest_path: Path) -> PluginManifest:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Invalid manifest format in {manifest_path}: expected a YAML mapping")

    for field_name in REQUIRED_FIELDS:
        if field_name not in data:
            raise ValueError(f"Missing required field '{field_name}' in {manifest_path}")

    return PluginManifest(
        name=str(data["name"]),
        version=str(data["version"]),
        entry_point=str(data["entry_point"]),
        description=str(data.get("description", "")),
        author=str(data.get("author", "")),
        brainapi_version=str(data.get("brainapi_version", "")),
        priority=int(data.get("priority", 100)),
        pip_dependencies=list(data.get("pip_dependencies", [])),
        tags=[str(t) for t in data.get("tags", [])],
        plugin_dir=manifest_path.parent,
    )
