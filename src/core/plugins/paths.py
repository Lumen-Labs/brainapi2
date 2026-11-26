from __future__ import annotations

import os
from pathlib import Path


def plugin_search_dirs(project_root: Path | None = None) -> list[Path]:
    root = project_root or Path(".")
    primary = Path(os.getenv("PLUGINS_DIR", str(root / "plugins")))
    extra_raw = os.getenv("PLUGINS_EXTRA_DIR", "").strip()
    dirs = [primary]
    if extra_raw:
        extra = Path(extra_raw)
        if extra.resolve() != primary.resolve():
            dirs.append(extra)
    return dirs


def install_plugins_dir(project_root: Path | None = None) -> Path:
    extra = os.getenv("PLUGINS_EXTRA_DIR", "").strip()
    if extra:
        return Path(extra)
    return plugin_search_dirs(project_root)[0]
