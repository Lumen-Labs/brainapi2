from __future__ import annotations

from typing import Any, Callable

_HOOKS: list[Callable[[Any], None]] = []


def register_ingest_commit_hook(fn: Callable[[Any], None]) -> None:
    if fn not in _HOOKS:
        _HOOKS.append(fn)


def unregister_ingest_commit_hook(fn: Callable[[Any], None]) -> None:
    try:
        _HOOKS.remove(fn)
    except ValueError:
        pass


def run_ingest_commit_hooks(payload: Any) -> None:
    for hook in list(_HOOKS):
        hook(payload)
