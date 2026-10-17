"""
File: /_provisioning.py
Project: postgresql

Process-wide Postgres infrastructure shared by the data / vectors / networkx
clients:

  * `get_pool(dbname)`               — LRU cache of psycopg2 ThreadedConnectionPools.
                                       Fork-safe (resets on child process).
  * `get_system_pool()`              — convenience helper for the registry DB
                                       (config.postgresql.system_database).
  * `get_brain_pool(brain_id)`       — convenience helper for the per-brain DB,
                                       resolved through `_naming.brain_db_name`.
  * `ensure_brain_database(brain_id)` — `CREATE DATABASE brain_<id>` if missing,
                                       executed against the maintenance database
                                       (default `postgres`).

Each brain gets its own Postgres database. The LRU keeps at most
`_MAX_OPEN_POOLS` pools warm; older pools are closed to bound Postgres's
`max_connections` footprint when there are many brains.
"""

from __future__ import annotations

import logging
import os
import threading
from collections import OrderedDict
from contextlib import contextmanager
from typing import Iterator, Optional

import psycopg2
import psycopg2.errors
import psycopg2.pool
from psycopg2 import sql

from src.config import config

from ._naming import BRAIN_DB_PREFIX, brain_db_name

logger = logging.getLogger(__name__)


_MAX_OPEN_POOLS = 16
_DEFAULT_MIN_CONN = 1
# Celery uses a threaded pool; keep enough connections for concurrent
# brain reads/writes without blocking behind the in-memory graph lock.
_DEFAULT_MAX_CONN = max(16, int(os.getenv("CELERY_WORKER_CONCURRENCY") or 4) * 4)
_MAINTENANCE_FALLBACK_DB = "postgres"


def _maintenance_dbname() -> str:
    return getattr(config.postgresql, "maintenance_database", None) or _MAINTENANCE_FALLBACK_DB


def _system_dbname() -> str:
    return getattr(config.postgresql, "system_database", None) or config.postgresql.database


class _PoolRegistry:
    def __init__(self) -> None:
        self._pools: "OrderedDict[str, psycopg2.pool.ThreadedConnectionPool]" = OrderedDict()
        self._lock = threading.RLock()
        self._pid: Optional[int] = None

    def _reset_if_forked(self) -> None:
        current = os.getpid()
        if self._pid is not None and self._pid != current:
            for name, pool in list(self._pools.items()):
                try:
                    pool.closeall()
                except Exception as exc:
                    logger.debug("Failed to close pool %s on fork: %s", name, exc)
            self._pools.clear()
        self._pid = current

    def get(
        self,
        dbname: str,
        minconn: int = _DEFAULT_MIN_CONN,
        maxconn: int = _DEFAULT_MAX_CONN,
    ) -> psycopg2.pool.ThreadedConnectionPool:
        with self._lock:
            self._reset_if_forked()
            existing = self._pools.get(dbname)
            if existing is not None:
                self._pools.move_to_end(dbname)
                return existing

            config.postgresql.validate_credentials()
            pool = psycopg2.pool.ThreadedConnectionPool(
                minconn=minconn,
                maxconn=maxconn,
                host=config.postgresql.host,
                port=config.postgresql.port,
                user=config.postgresql.username,
                password=config.postgresql.password,
                dbname=dbname,
            )
            self._pools[dbname] = pool
            self._evict_if_needed()
            return pool

    def _evict_if_needed(self) -> None:
        while len(self._pools) > _MAX_OPEN_POOLS:
            evicted_name, evicted_pool = self._pools.popitem(last=False)
            try:
                evicted_pool.closeall()
            except Exception as exc:
                logger.warning("Failed to close LRU pool %s: %s", evicted_name, exc)
            else:
                logger.debug("Evicted LRU pool: %s", evicted_name)

    def close(self, dbname: str) -> None:
        with self._lock:
            pool = self._pools.pop(dbname, None)
        if pool is None:
            return
        try:
            pool.closeall()
        except Exception as exc:
            logger.debug("Failed to close pool %s: %s", dbname, exc)


_registry = _PoolRegistry()


def get_pool(dbname: str, **kwargs) -> psycopg2.pool.ThreadedConnectionPool:
    return _registry.get(dbname, **kwargs)


def get_system_pool() -> psycopg2.pool.ThreadedConnectionPool:
    return _registry.get(_system_dbname())


def get_brain_pool(brain_id: str) -> psycopg2.pool.ThreadedConnectionPool:
    return _registry.get(brain_db_name(brain_id))


@contextmanager
def borrow(pool: psycopg2.pool.ThreadedConnectionPool) -> Iterator[psycopg2.extensions.connection]:
    conn = pool.getconn()
    try:
        yield conn
    finally:
        pool.putconn(conn)


_created_dbs: set[str] = set()
_create_lock = threading.Lock()


def ensure_database_exists(dbname: str) -> None:
    """Create `dbname` if it does not yet exist on the cluster.

    Idempotent and safe under concurrent callers. Uses the maintenance database
    (typically `postgres`) for the CREATE DATABASE statement and records a
    process-local cache so future calls become free.
    """
    if dbname in _created_dbs:
        return
    with _create_lock:
        if dbname in _created_dbs:
            return
        conn = psycopg2.connect(
            host=config.postgresql.host,
            port=config.postgresql.port,
            user=config.postgresql.username,
            password=config.postgresql.password,
            dbname=_maintenance_dbname(),
        )
        try:
            conn.autocommit = True
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT 1 FROM pg_database WHERE datname = %s",
                    (dbname,),
                )
                if cur.fetchone() is None:
                    try:
                        cur.execute(
                            sql.SQL("CREATE DATABASE {}").format(sql.Identifier(dbname))
                        )
                        logger.info("Created Postgres database: %s", dbname)
                    except psycopg2.errors.DuplicateDatabase:
                        pass
        finally:
            conn.close()
        _created_dbs.add(dbname)


def ensure_brain_database(brain_id: str) -> str:
    """Return the per-brain database name, creating the database if needed."""
    dbname = brain_db_name(brain_id)
    ensure_database_exists(dbname)
    return dbname


def ensure_system_database() -> str:
    dbname = _system_dbname()
    ensure_database_exists(dbname)
    return dbname


def list_brain_database_names() -> list[str]:
    # LIKE '_' is a single-char wildcard; escape so prefix "brain_" is literal.
    # Otherwise "brainapi" (system DB) matches "brain_%".
    escaped_prefix = (
        BRAIN_DB_PREFIX.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    )
    system_db = _system_dbname()
    with borrow(_registry.get(_maintenance_dbname())) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT datname FROM pg_database "
                "WHERE datname LIKE %s ESCAPE %s ORDER BY datname",
                (f"{escaped_prefix}%", "\\"),
            )
            return [
                row[0]
                for row in cur.fetchall()
                if row[0] != system_db
            ]


def clear_brain_database(
    brain_id: str,
    *,
    table_prefixes: tuple[str, ...] | None = None,
) -> list[str]:
    """Truncate every application table in one brain database.

    The database and schemas remain in place so API and worker connection pools
    stay valid. This makes an online clear safe across processes while removing
    data, graph records, vectors, and plugin-owned tables in the public schema.
    """
    dbname = brain_db_name(brain_id)
    maintenance = _registry.get(_maintenance_dbname())
    with borrow(maintenance) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (dbname,))
            if cur.fetchone() is None:
                return []

    with borrow(get_brain_pool(brain_id)) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT tablename FROM pg_tables "
                "WHERE schemaname = 'public' ORDER BY tablename"
            )
            tables = [
                row[0]
                for row in cur.fetchall()
                if table_prefixes is None
                or row[0].startswith(table_prefixes)
            ]
            if tables:
                identifiers = sql.SQL(", ").join(
                    sql.Identifier(table) for table in tables
                )
                cur.execute(
                    sql.SQL("TRUNCATE TABLE {} RESTART IDENTITY CASCADE").format(
                        identifiers
                    )
                )
        conn.commit()
    return tables


def drop_brain_database(brain_id: str) -> bool:
    dbname = brain_db_name(brain_id)
    _registry.close(dbname)
    _created_dbs.discard(dbname)
    conn = psycopg2.connect(
        host=config.postgresql.host,
        port=config.postgresql.port,
        user=config.postgresql.username,
        password=config.postgresql.password,
        dbname=_maintenance_dbname(),
    )
    try:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (dbname,))
            if cur.fetchone() is None:
                return False
            cur.execute(
                "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
                "WHERE datname = %s AND pid <> pg_backend_pid()",
                (dbname,),
            )
            cur.execute(
                sql.SQL("DROP DATABASE IF EXISTS {}").format(sql.Identifier(dbname))
            )
            logger.info("Dropped Postgres database: %s", dbname)
            return True
    finally:
        conn.close()
