from __future__ import annotations

import logging
import os
import tarfile
import tempfile
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import httpx

logger = logging.getLogger("brainapi.plugins.registry")


@dataclass
class PluginMeta:
    name: str
    version: str
    description: str = ""
    author: str = ""
    versions: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PluginMeta":
        return cls(
            name=data.get("name", ""),
            version=data.get("version", ""),
            description=data.get("description", ""),
            author=data.get("author", ""),
            versions=data.get("versions", []),
        )


class PluginRegistryClient:
    def __init__(
        self,
        registry_url: str,
        publisher_id: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = 60.0,
    ):
        self.registry_url = registry_url.rstrip("/")
        self.publisher_id = publisher_id
        self.api_key = api_key
        self.timeout = timeout

    def _headers(self, with_publisher: bool = False) -> dict[str, str]:
        headers: dict[str, str] = {"Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if with_publisher and self.publisher_id:
            headers["X-Publisher-Id"] = self.publisher_id
        return headers

    def register_publisher(self, publisher_id: Optional[str] = None) -> dict[str, Any]:
        payload = {}
        if publisher_id:
            payload["publisher_id"] = publisher_id
        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(self._url("/publishers"), json=payload)
            response.raise_for_status()
            return response.json()

    def _url(self, path: str) -> str:
        return f"{self.registry_url}/{path.lstrip('/')}"

    def list_plugins(self) -> list[PluginMeta]:
        with httpx.Client(timeout=self.timeout) as client:
            response = client.get(self._url("/plugins"), headers=self._headers())
            response.raise_for_status()
            data = response.json()

        plugins = data if isinstance(data, list) else data.get("plugins", [])
        return [PluginMeta.from_dict(p) for p in plugins]

    def get_plugin_info(self, name: str) -> PluginMeta:
        with httpx.Client(timeout=self.timeout) as client:
            response = client.get(self._url(f"/plugins/{name}"), headers=self._headers())
            response.raise_for_status()
            return PluginMeta.from_dict(response.json())

    def download(self, name: str, version: str = "latest", target_dir: Optional[Path] = None) -> Path:
        url = self._url(f"/plugins/{name}/versions/{version}/download")

        with httpx.Client(timeout=self.timeout, follow_redirects=True) as client:
            response = client.get(url, headers=self._headers())
            response.raise_for_status()

        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
            tmp.write(response.content)
            tmp_path = Path(tmp.name)

        extract_dir = tempfile.mkdtemp()
        try:
            if target_dir is None:
                target_dir = Path("plugins")
            target_dir.mkdir(parents=True, exist_ok=True)

            with tarfile.open(tmp_path, "r:gz") as tar:
                tar.extractall(extract_dir)

            manifest_file = None
            for root, _dirs, files in os.walk(extract_dir):
                if "plugin.yaml" in files:
                    manifest_file = Path(root) / "plugin.yaml"
                    break

            if manifest_file is None:
                raise FileNotFoundError("plugin.yaml not found in archive")

            source_dir = manifest_file.parent

            plugin_dir = target_dir / name
            if plugin_dir.exists():
                shutil.rmtree(plugin_dir)
            shutil.copytree(source_dir, plugin_dir)

            logger.info("Downloaded plugin '%s' v%s to '%s'", name, version, plugin_dir)
            return plugin_dir
        finally:
            tmp_path.unlink(missing_ok=True)
            shutil.rmtree(extract_dir, ignore_errors=True)

    def publish(self, archive_path: Path, name: Optional[str] = None, force: bool = False) -> dict[str, Any]:
        url = self._url("/plugins")
        if force:
            url += "?force=true"
        headers = self._headers(with_publisher=True)
        headers.pop("Accept", None)

        with open(archive_path, "rb") as f:
            files = {"archive": (archive_path.name, f, "application/gzip")}
            data = {}
            if name:
                data["name"] = name

            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(url, headers=headers, files=files, data=data)
                response.raise_for_status()
                return response.json()

    def delete(self, name: str, version: Optional[str] = None) -> dict[str, Any]:
        path = f"/plugins/{name}/versions/{version}" if version else f"/plugins/{name}"
        with httpx.Client(timeout=self.timeout) as client:
            response = client.delete(self._url(path), headers=self._headers(with_publisher=True))
            response.raise_for_status()
            return response.json()
