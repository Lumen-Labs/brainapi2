from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path


def _load_module(plugin_dir: Path, relative_path: str, module_name: str):
    path = plugin_dir / relative_path
    existing = sys.modules.get(module_name)
    if existing is not None:
        existing_file = getattr(existing, "__file__", None)
        if existing_file and Path(existing_file).resolve() == path.resolve():
            return existing
        sys.modules.pop(module_name, None)
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


def discover_plugin_celery(plugins_dir: Path | list[Path]) -> tuple[tuple, dict, dict]:
    directories = plugins_dir if isinstance(plugins_dir, list) else [plugins_dir]
    queues_by_name: dict = {}
    routes: dict = {}
    beat: dict = {}

    for directory in directories:
        if not directory.exists():
            continue
        for child in sorted(directory.iterdir()):
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
                for queue in getattr(celery_mod, "QUEUES", ()) or ():
                    queues_by_name[getattr(queue, "name", id(queue))] = queue
                routes.update(getattr(celery_mod, "ROUTES", {}) or {})
                beat.update(getattr(celery_mod, "BEAT_SCHEDULE", {}) or {})
            finally:
                if added_plugin_path and plugin_dir_str in sys.path:
                    sys.path.remove(plugin_dir_str)
                sys.path_importer_cache.pop(plugin_dir_str, None)

    return tuple(queues_by_name.values()), routes, beat


def default_plugins_dir(project_root: Path) -> Path:
    return Path(os.getenv("PLUGINS_DIR", str(project_root / "plugins")))


def default_plugin_dirs(project_root: Path) -> list[Path]:
    from src.core.plugins.paths import plugin_search_dirs

    return plugin_search_dirs(project_root)
