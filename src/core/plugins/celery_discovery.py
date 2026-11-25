from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path


def _load_module(plugin_dir: Path, relative_path: str, module_name: str):
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    path = plugin_dir / relative_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


def discover_plugin_celery(plugins_dir: Path) -> tuple[tuple, dict, dict]:
    if not plugins_dir.exists():
        return (), {}, {}

    queues: list = []
    routes: dict = {}
    beat: dict = {}

    for child in sorted(plugins_dir.iterdir()):
        if not child.is_dir():
            continue
        celery_module_path = child / "workers" / "celery.py"
        tasks_module_path = child / "workers" / "tasks.py"
        if not celery_module_path.exists() or not tasks_module_path.exists():
            continue

        plugin_dir_str = str(child)
        added_plugin_path = plugin_dir_str not in sys.path
        if added_plugin_path:
            sys.path.insert(0, plugin_dir_str)
        try:
            safe_name = child.name.replace("-", "_")
            celery_mod = _load_module(
                child,
                "workers/celery.py",
                f"brainapi_plugin_{safe_name}_workers_celery",
            )
            _load_module(
                child,
                "workers/tasks.py",
                f"brainapi_plugin_{safe_name}_workers_tasks",
            )
            queues.extend(getattr(celery_mod, "QUEUES", ()))
            routes.update(getattr(celery_mod, "ROUTES", {}))
            beat.update(getattr(celery_mod, "BEAT_SCHEDULE", {}) or {})
        finally:
            if added_plugin_path and plugin_dir_str in sys.path:
                sys.path.remove(plugin_dir_str)
            sys.path_importer_cache.pop(plugin_dir_str, None)

    return tuple(queues), routes, beat


def default_plugins_dir(project_root: Path) -> Path:
    return Path(os.getenv("PLUGINS_DIR", str(project_root / "plugins")))
