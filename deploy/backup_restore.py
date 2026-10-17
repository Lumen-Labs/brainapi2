#!/usr/bin/env python3
"""Offline-consistent BrainAPI profile backup, verification, and restore."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import stat
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import BinaryIO


ROOT = Path(__file__).resolve().parents[1]
DEPLOY = ROOT / "deploy"
ARCHIVER_IMAGE = "alpine:3.21.3"
APP_SERVICES = ("nginx", "brainapi", "brainapi-mcp", "brainapi-worker")
PROFILE_VOLUMES = {
    "light": ("redis-data", "brainapi-plugins"),
    "heavy": (
        "redis-data",
        "milvus-data",
        "etcd-data",
        "minio-data",
        "brainapi-plugins",
    ),
}
PROFILE_ALL_VOLUMES = {
    "light": ("redis-data", "postgres-data", "brainapi-plugins"),
    "heavy": (
        "redis-data",
        "neo4j-data",
        "neo4j-logs",
        "neo4j-plugins",
        "etcd-data",
        "minio-data",
        "milvus-data",
        "mongo-data",
        "mongo-config",
        "brainapi-plugins",
    ),
}
RESTORE_ORDER = {
    "light": ("redis-data", "postgres", "brainapi-plugins", "stack"),
    "heavy": (
        "redis-data",
        "etcd-data",
        "minio-data",
        "milvus-data",
        "neo4j",
        "mongo",
        "brainapi-plugins",
        "stack",
    ),
}


def _load_env(path: Path) -> dict[str, str]:
    values = dict(os.environ)
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values.setdefault(key.strip(), value.strip().strip("'\""))
    return values


class Stack:
    def __init__(
        self,
        profile: str,
        env_file: Path,
        project: str,
        dry_run: bool,
        compose_files: list[Path] | None = None,
    ):
        self.profile = profile
        self.env_file = env_file.resolve()
        self.project = project
        self.dry_run = dry_run
        self.compose_files = [DEPLOY / f"docker-compose.{profile}.yaml"]
        self.compose_files.extend(path.resolve() for path in (compose_files or []))
        self.env = _load_env(self.env_file)
        self.env["BRAINAPI_ENV_FILE"] = str(self.env_file)
        self.base = [
            "docker",
            "compose",
            "--project-name",
            project,
            "--env-file",
            str(self.env_file),
        ]
        for compose_file in self.compose_files:
            self.base.extend(["-f", str(compose_file)])

    def run(
        self,
        args: list[str],
        *,
        stdout: BinaryIO | int | None = None,
        stdin: BinaryIO | int | None = None,
        capture: bool = False,
        check: bool = True,
    ) -> subprocess.CompletedProcess:
        command = [*self.base, *args]
        if self.dry_run:
            print("+", shlex.join(command))
            return subprocess.CompletedProcess(command, 0, stdout="" if capture else None)
        return subprocess.run(
            command,
            env=self.env,
            stdin=stdin,
            stdout=subprocess.PIPE if capture else stdout,
            stderr=subprocess.PIPE if capture else None,
            text=capture,
            check=check,
        )

    def docker(
        self,
        args: list[str],
        *,
        stdout: BinaryIO | int | None = None,
        stdin: BinaryIO | int | None = None,
        capture: bool = False,
        check: bool = True,
    ) -> subprocess.CompletedProcess:
        command = ["docker", *args]
        if self.dry_run:
            print("+", shlex.join(command))
            return subprocess.CompletedProcess(command, 0, stdout="" if capture else None)
        return subprocess.run(
            command,
            env=self.env,
            stdin=stdin,
            stdout=subprocess.PIPE if capture else stdout,
            stderr=subprocess.PIPE if capture else None,
            text=capture,
            check=check,
        )

    def volume_name(self, key: str) -> str:
        return f"{self.project}_{key}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_stream(stack: Stack, args: list[str], destination: Path) -> None:
    if stack.dry_run:
        stack.run(args)
        destination.touch()
        return
    with destination.open("wb") as output:
        stack.run(args, stdout=output)


def _dump_postgres_brain_databases(stack: Stack, destination: Path) -> None:
    script = r'''
set -eu
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
psql -U "$POSTGRES_USER" -d postgres -Atc \
  "SELECT datname FROM pg_database WHERE datistemplate = false AND left(datname, 6) = 'brain_' ORDER BY datname" |
while IFS= read -r database; do
  test -n "$database" || continue
  pg_dump -U "$POSTGRES_USER" -d "$database" --format=custom --file="$tmp/$database.dump"
done
tar -C "$tmp" -czf - .
'''.strip()
    _write_stream(
        stack,
        ["exec", "-T", "postgres", "sh", "-c", script],
        destination,
    )


def _restore_postgres_brain_databases(
    stack: Stack, archive: Path, pg_user: str
) -> None:
    with tempfile.TemporaryDirectory(prefix="brainapi-postgres-restore-") as tmp:
        tmp_path = Path(tmp)
        if not stack.dry_run:
            subprocess.run(
                ["tar", "-C", str(tmp_path), "-xzf", str(archive)], check=True
            )
        dumps = sorted(tmp_path.glob("brain_*.dump"))
        if stack.dry_run:
            dumps = [tmp_path / "brain_example.dump"]
        for dump in dumps:
            database = dump.stem
            if not database.replace("_", "").isalnum() or not database.startswith(
                "brain_"
            ):
                raise RuntimeError(f"Invalid PostgreSQL brain dump name: {dump.name}")
            stack.run(
                ["exec", "-T", "postgres", "createdb", "-U", pg_user, database],
                check=False,
            )
            if stack.dry_run:
                stack.run(
                    [
                        "exec",
                        "-T",
                        "postgres",
                        "pg_restore",
                        "-U",
                        pg_user,
                        "-d",
                        database,
                        "--clean",
                        "--if-exists",
                    ]
                )
            else:
                with dump.open("rb") as source:
                    stack.run(
                        [
                            "exec",
                            "-T",
                            "postgres",
                            "pg_restore",
                            "-U",
                            pg_user,
                            "-d",
                            database,
                            "--clean",
                            "--if-exists",
                        ],
                        stdin=source,
                    )


def _archive_volume(stack: Stack, key: str, destination: Path) -> None:
    if stack.dry_run:
        stack.docker(["run", "--rm", "-v", f"{stack.volume_name(key)}:/source:ro", ARCHIVER_IMAGE, "tar", "-C", "/source", "-czf", "-", "."])
        destination.touch()
        return
    with destination.open("wb") as output:
        stack.docker(
            [
                "run",
                "--rm",
                "-v",
                f"{stack.volume_name(key)}:/source:ro",
                ARCHIVER_IMAGE,
                "tar",
                "-C",
                "/source",
                "-czf",
                "-",
                ".",
            ],
            stdout=output,
        )


def _restore_volume(stack: Stack, key: str, archive: Path) -> None:
    if stack.dry_run:
        stack.docker(["run", "--rm", "-i", "-v", f"{stack.volume_name(key)}:/target", ARCHIVER_IMAGE, "tar", "-C", "/target", "-xzf", "-"])
        return
    with archive.open("rb") as source:
        stack.docker(
            [
                "run",
                "--rm",
                "-i",
                "-v",
                f"{stack.volume_name(key)}:/target",
                ARCHIVER_IMAGE,
                "tar",
                "-C",
                "/target",
                "-xzf",
                "-",
            ],
            stdin=source,
        )


def _running_services(stack: Stack) -> list[str]:
    result = stack.run(
        ["ps", "--services", "--filter", "status=running"], capture=True
    )
    return [line for line in (result.stdout or "").splitlines() if line]


def _current_image(stack: Stack) -> str:
    result = stack.run(["config", "--format", "json"], capture=True)
    if stack.dry_run:
        return stack.env.get("BRAINAPI_IMAGE", "dry-run")
    config = json.loads(result.stdout)
    return str(config["services"]["brainapi"]["image"])


def _manifest(backup_dir: Path) -> dict:
    return json.loads((backup_dir / "manifest.json").read_text(encoding="utf-8"))


def verify_backup(backup_dir: Path, expected_profile: str | None = None) -> dict:
    manifest = _manifest(backup_dir)
    if manifest.get("format_version") != 1:
        raise RuntimeError("Unsupported backup manifest format")
    if expected_profile and manifest.get("profile") != expected_profile:
        raise RuntimeError("Backup profile does not match the requested profile")
    for name, expected in manifest.get("checksums", {}).items():
        path = backup_dir / name
        if not path.is_file():
            raise RuntimeError(f"Missing backup artifact: {name}")
        actual = _sha256(path)
        if actual != expected:
            raise RuntimeError(f"Checksum mismatch: {name}")
    return manifest


def backup(stack: Stack, backup_root: Path) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_root = backup_root.expanduser().resolve()
    backup_root.mkdir(parents=True, mode=0o700, exist_ok=True)
    backup_root.chmod(stat.S_IRWXU)
    destination = backup_root / f"{stack.profile}-{timestamp}"
    destination.mkdir(parents=True, mode=0o700, exist_ok=False)
    destination.chmod(stat.S_IRWXU)
    running = _running_services(stack)

    try:
        stack.run(["stop", *APP_SERVICES])
        if stack.profile == "light":
            pg_user = stack.env.get("POSTGRES_USERNAME", "brainapi")
            pg_db = stack.env.get("POSTGRES_SYSTEM_DATABASE", "brainapi")
            _write_stream(
                stack,
                ["exec", "-T", "postgres", "pg_dump", "-U", pg_user, "-d", pg_db, "--format=custom"],
                destination / "postgres.dump",
            )
            _dump_postgres_brain_databases(
                stack, destination / "postgres-brains.tar.gz"
            )
            stack.run(["exec", "-T", "redis", "sh", "-c", 'redis-cli -a "$REDIS_PASSWORD" SAVE'])
            stack.run(["stop", "redis", "postgres"])
        else:
            mongo_user = stack.env.get("MONGO_USERNAME", "brainapi")
            _write_stream(
                stack,
                ["exec", "-T", "mongo", "sh", "-c", f'mongodump --archive --gzip --username {shlex.quote(mongo_user)} --password "$MONGO_INITDB_ROOT_PASSWORD" --authenticationDatabase admin'],
                destination / "mongo.archive.gz",
            )
            stack.run(["exec", "-T", "redis", "sh", "-c", 'redis-cli -a "$REDIS_PASSWORD" SAVE'])
            stack.run(["stop", "neo4j", "mongo", "milvus", "etcd", "minio", "redis"])
            neo_dir = destination / "neo4j"
            neo_dir.mkdir(mode=0o700)
            for database in ("system", "neo4j"):
                stack.run(
                    ["run", "--rm", "--no-deps", "-v", f"{neo_dir}:/backup", "neo4j", "neo4j-admin", "database", "dump", database, "--to-path=/backup", "--overwrite-destination=true"]
                )
            if not stack.dry_run:
                subprocess.run(["tar", "-C", str(neo_dir), "-czf", str(destination / "neo4j.dumps.tar.gz"), "."], check=True)
                for child in neo_dir.iterdir():
                    child.unlink()
                neo_dir.rmdir()

        for volume in PROFILE_VOLUMES[stack.profile]:
            _archive_volume(stack, volume, destination / f"{volume}.tar.gz")

        artifacts = sorted(path for path in destination.iterdir() if path.is_file())
        manifest = {
            "format_version": 1,
            "profile": stack.profile,
            "brainapi_image": _current_image(stack),
            "brainapi_version": _current_image(stack).rsplit(":", 1)[-1],
            "timestamp": timestamp,
            "services": (
                ["postgres", "redis", "brainapi-plugins"]
                if stack.profile == "light"
                else ["neo4j", "mongo", "redis", "milvus", "etcd", "minio", "brainapi-plugins"]
            ),
            "restore_order": list(RESTORE_ORDER[stack.profile]),
            "checksums": {path.name: _sha256(path) for path in artifacts},
        }
        (destination / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        verify_backup(destination, stack.profile)
        return destination
    finally:
        if running:
            stack.run(["start", *running], check=False)


def _assert_empty_stopped(stack: Stack) -> None:
    running = _running_services(stack)
    if running:
        raise RuntimeError("Restore target must be stopped; running: " + ", ".join(running))
    for key in PROFILE_ALL_VOLUMES[stack.profile]:
        volume = stack.volume_name(key)
        exists = stack.docker(["volume", "inspect", volume], check=False).returncode == 0
        if not exists or stack.dry_run:
            continue
        result = stack.docker(
            ["run", "--rm", "-v", f"{volume}:/source:ro", ARCHIVER_IMAGE, "sh", "-c", "test -z \"$(find /source -mindepth 1 -maxdepth 1 -print -quit)\""],
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Restore target volume is not empty: {volume}")


def restore(stack: Stack, backup_dir: Path) -> None:
    manifest = verify_backup(backup_dir, stack.profile)
    if manifest["brainapi_image"] != _current_image(stack):
        raise RuntimeError("Backup image does not match BRAINAPI_IMAGE")
    _assert_empty_stopped(stack)
    stack.run(["up", "--no-start"])

    for key in PROFILE_VOLUMES[stack.profile]:
        _restore_volume(stack, key, backup_dir / f"{key}.tar.gz")

    if stack.profile == "light":
        stack.run(["up", "-d", "--wait", "postgres"])
        pg_user = stack.env.get("POSTGRES_USERNAME", "brainapi")
        pg_db = stack.env.get("POSTGRES_SYSTEM_DATABASE", "brainapi")
        if stack.dry_run:
            stack.run(["exec", "-T", "postgres", "pg_restore", "-U", pg_user, "-d", pg_db, "--clean", "--if-exists"])
        else:
            with (backup_dir / "postgres.dump").open("rb") as source:
                stack.run(["exec", "-T", "postgres", "pg_restore", "-U", pg_user, "-d", pg_db, "--clean", "--if-exists"], stdin=source)
        _restore_postgres_brain_databases(
            stack, backup_dir / "postgres-brains.tar.gz", pg_user
        )
    else:
        with tempfile.TemporaryDirectory(prefix="brainapi-neo4j-restore-") as tmp:
            tmp_path = Path(tmp)
            if not stack.dry_run:
                subprocess.run(["tar", "-C", str(tmp_path), "-xzf", str(backup_dir / "neo4j.dumps.tar.gz")], check=True)
            for database in ("system", "neo4j"):
                stack.run(["run", "--rm", "--no-deps", "-v", f"{tmp_path}:/backup:ro", "neo4j", "neo4j-admin", "database", "load", database, "--from-path=/backup", "--overwrite-destination=true"])
        stack.run(["up", "-d", "--wait", "mongo"])
        mongo_user = stack.env.get("MONGO_USERNAME", "brainapi")
        if stack.dry_run:
            stack.run(["exec", "-T", "mongo", "sh", "-c", f'mongorestore --archive --gzip --drop --username {shlex.quote(mongo_user)} --password "$MONGO_INITDB_ROOT_PASSWORD" --authenticationDatabase admin'])
        else:
            with (backup_dir / "mongo.archive.gz").open("rb") as source:
                stack.run(["exec", "-T", "mongo", "sh", "-c", f'mongorestore --archive --gzip --drop --username {shlex.quote(mongo_user)} --password "$MONGO_INITDB_ROOT_PASSWORD" --authenticationDatabase admin'], stdin=source)

    stack.run(["up", "-d", "--wait"])


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("backup", "verify", "restore"))
    parser.add_argument("--profile", choices=("light", "heavy"), required=True)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--project-name", default="brainapi")
    parser.add_argument("--backup-dir", type=Path, default=Path(os.getenv("BACKUP_DIR", "/srv/brainapi/backups")))
    parser.add_argument("--archive", type=Path, help="Backup directory for verify/restore")
    parser.add_argument(
        "--compose-file",
        action="append",
        default=[],
        type=Path,
        help="Additional Compose override file; may be repeated.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        if args.command == "verify":
            if not args.archive:
                raise RuntimeError("--archive is required for verify")
            verify_backup(args.archive.resolve(), args.profile)
            print(f"Verified {args.archive}")
            return 0
        stack = Stack(
            args.profile,
            args.env_file,
            args.project_name,
            args.dry_run,
            args.compose_file,
        )
        if args.command == "backup":
            destination = backup(stack, args.backup_dir)
            print(destination)
        else:
            if not args.archive:
                raise RuntimeError("--archive is required for restore")
            restore(stack, args.archive.resolve())
            print(f"Restored {args.archive}")
        return 0
    except (KeyError, OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"backup_restore: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
