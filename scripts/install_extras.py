"""
Install only the pyproject.toml optional-dependency groups that the active
.env requires.

Usage (no TUI required):

    python scripts/install_extras.py                    # auto-detect from .env
    python scripts/install_extras.py docling-ocr        # force a specific group
    python scripts/install_extras.py --all              # install every optional group
    python scripts/install_extras.py --dry-run          # print pip command, don't run
    python scripts/install_extras.py --check            # exit non-zero if missing

Reads ./.env (or $BRAINAPI_DOTENV) for OCR_MODE and any future toggles, then
calls `python -m pip install` against the active interpreter with the matching
specs from pyproject.toml's [project.optional-dependencies].

Safe to re-run: pip is a no-op when the specs are already satisfied.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import re
import subprocess
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore[no-redef]


REPO_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT_PATH = REPO_ROOT / "pyproject.toml"
DEFAULT_ENV_PATH = Path(os.environ.get("BRAINAPI_DOTENV", REPO_ROOT / ".env"))


def parse_dotenv(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    result: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        result[key.strip()] = value.strip().strip('"').strip("'")
    return result


def _load_pyproject() -> dict:
    if not PYPROJECT_PATH.exists():
        return {}
    with PYPROJECT_PATH.open("rb") as fh:
        return tomllib.load(fh)


def _poetry_dep_to_spec(name: str, value) -> str | None:
    if name == "python":
        return None
    if isinstance(value, str):
        return name if value == "*" else f"{name}{value}"
    if isinstance(value, dict):
        version = value.get("version", "")
        extras = value.get("extras", []) or []
        head = name + (f"[{','.join(extras)}]" if extras else "")
        if not version or version == "*":
            return head
        return f"{head}{version}"
    return None


def load_base_deps() -> list[str]:
    data = _load_pyproject()
    project_deps = data.get("project", {}).get("dependencies", []) or []
    if project_deps:
        return list(project_deps)
    poetry_deps = (
        data.get("tool", {}).get("poetry", {}).get("dependencies", {}) or {}
    )
    out: list[str] = []
    for name, value in poetry_deps.items():
        spec = _poetry_dep_to_spec(name, value)
        if spec:
            out.append(spec)
    return out


def load_optional_groups() -> dict[str, list[str]]:
    data = _load_pyproject()
    project_optional = (
        data.get("project", {}).get("optional-dependencies", {}) or {}
    )
    if project_optional:
        return {k: list(v) for k, v in project_optional.items()}
    poetry_groups = data.get("tool", {}).get("poetry", {}).get("group", {}) or {}
    out: dict[str, list[str]] = {}
    for group_name, group_value in poetry_groups.items():
        if not isinstance(group_value, dict):
            continue
        deps = group_value.get("dependencies", {}) or {}
        specs: list[str] = []
        for name, value in deps.items():
            spec = _poetry_dep_to_spec(name, value)
            if spec:
                specs.append(spec)
        if specs:
            out[group_name] = specs
    return out


POSTGRESQL_BACKEND_SPECS = [
    "psycopg2-binary>=2.9.6,<3.0.0",
    "pgvector>=0.3.0,<1.0.0",
    "networkx>=3.0,<4.0.0",
]


def uses_postgresql_backend(env: dict[str, str]) -> bool:
    vector = env.get("VECTOR_DB", "milvus").lower()
    data = env.get("DATA_DB", "mongo").lower()
    graph = env.get("GRAPH_DB", "neo4j").lower()
    return vector == "postgresql" or data == "postgresql" or graph == "networkx"


def required_groups(env: dict[str, str]) -> list[str]:
    groups: list[str] = []
    if env.get("OCR_MODE", "docparser").lower() == "docling":
        groups.append("docling-ocr")
    if uses_postgresql_backend(env):
        groups.append("postgresql-backend")
    return groups


def required_backend_specs(env: dict[str, str]) -> list[str]:
    if not uses_postgresql_backend(env):
        return []
    optional = load_optional_groups()
    if "postgresql-backend" in optional:
        return []
    return list(POSTGRESQL_BACKEND_SPECS)


def package_name_from_spec(spec: str) -> str:
    spec = spec.strip()
    match = re.match(r"^([A-Za-z0-9][A-Za-z0-9._-]*)", spec)
    if match:
        return match.group(1)
    return spec


def package_installed(name: str) -> bool:
    pkg = package_name_from_spec(name)
    try:
        from importlib.metadata import PackageNotFoundError, version

        version(pkg)
        return True
    except PackageNotFoundError:
        pass
    except Exception:
        pass
    import_names = {pkg.replace("-", "_"), pkg}
    if pkg == "psycopg2-binary":
        import_names.add("psycopg2")
    return any(importlib.util.find_spec(c) is not None for c in import_names)


def filter_missing(specs: list[str]) -> list[str]:
    return [s for s in specs if not package_installed(package_name_from_spec(s))]


def install_specs(specs: list[str], dry_run: bool) -> int:
    if not specs:
        return 0
    cmd = [sys.executable, "-m", "pip", "install", *specs]
    print(f"$ {' '.join(cmd)}")
    if dry_run:
        return 0
    return subprocess.call(cmd)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "groups",
        nargs="*",
        help="Optional groups to install. Omit to auto-detect from .env.",
    )
    parser.add_argument(
        "--env-file",
        default=str(DEFAULT_ENV_PATH),
        help="Path to .env (default: ./.env, or $BRAINAPI_DOTENV).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the pip command without running it.",
    )
    parser.add_argument(
        "--check", action="store_true",
        help="Exit non-zero if any required extras are missing; do not install.",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Install every optional group declared in pyproject.toml.",
    )
    parser.add_argument(
        "--include-base", action="store_true",
        help="Also install [project].dependencies / [tool.poetry.dependencies] base deps.",
    )
    parser.add_argument(
        "--upgrade-pip", action="store_true",
        help="Run `pip install --upgrade pip` before installing anything.",
    )
    args = parser.parse_args(argv)

    optional_groups = load_optional_groups()

    if args.all:
        wanted = list(optional_groups.keys())
    elif args.groups:
        wanted = args.groups
    else:
        env = parse_dotenv(Path(args.env_file))
        wanted = required_groups(env)
        if wanted:
            print(f"Auto-detected required groups from {args.env_file}: {wanted}")
        elif not args.include_base:
            print(
                f"No optional groups required by {args.env_file} "
                f"(OCR_MODE={env.get('OCR_MODE', 'docparser')}).",
            )

    unknown = [g for g in wanted if g not in optional_groups]
    known = [g for g in wanted if g in optional_groups]
    if unknown:
        print(
            f"Skipping optional groups not declared in pyproject.toml: {unknown}",
            file=sys.stderr,
        )
        if optional_groups:
            print(
                f"Available optional groups: {list(optional_groups.keys())}",
                file=sys.stderr,
            )
        else:
            print(
                "No optional-dependency groups are declared in pyproject.toml. "
                "These packages may already be part of base dependencies — "
                "continuing without them.",
                file=sys.stderr,
            )

    wanted = known
    extra_specs: list[str] = []
    for g in wanted:
        extra_specs.extend(optional_groups[g])

    env = parse_dotenv(Path(args.env_file))
    backend_specs = (
        required_backend_specs(env) if not args.groups and not args.all else []
    )

    if args.check:
        missing = filter_missing([*extra_specs, *backend_specs])
        if missing:
            print(f"Missing extras: {missing}", file=sys.stderr)
            hint_groups = wanted or (
                ["postgresql-backend"] if backend_specs else []
            )
            if hint_groups:
                print(
                    f"Install them with: {sys.executable} "
                    f"scripts/install_extras.py {' '.join(hint_groups)}",
                    file=sys.stderr,
                )
            return 1
        print("All required extras are installed.")
        return 0

    base_specs = load_base_deps() if args.include_base else []
    all_specs = [*base_specs, *extra_specs, *backend_specs]

    if not all_specs:
        return 0

    if args.upgrade_pip and not args.dry_run:
        subprocess.call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])

    return install_specs(all_specs, args.dry_run)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
