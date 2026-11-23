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
`POSTGRES_POOL_MAX_OPEN` pools warm; older idle pools are closed to bound
Postgres's `max_connections` footprint when there are many brains.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections import OrderedDict
from contextlib import contextmanager
from typing import Iterator, Optional

import psycopg2
import psycopg2.errors
import psycopg2.pool
from psycopg2 import sql
from psycopg2.extensions import TRANSACTION_STATUS_IDLE

from src.config import config

from ._naming import BRAIN_DB_PREFIX, brain_db_name

logger = logging.getLogger(__name__)


_MAINTENANCE_FALLBACK_DB = "postgres"
_DEFAULT_POOL_TIMEOUT = 5.0


class PoolTimeout(Exception):
    """Raised when the per-brain pool has no free connection before the wait limit."""


def _env_int(name: str, default: int, *, minimum: int) -> int:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        return max(minimum, int(raw))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float, *, minimum: float) -> float:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        return max(minimum, float(raw))
    except (TypeError, ValueError):
        return default


def _max_open_pools() -> int:
    return _env_int("POSTGRES_POOL_MAX_OPEN", 16, minimum=1)


def _min_conn() -> int:
    return _env_int("POSTGRES_POOL_MIN_CONN", 1, minimum=0)


def _default_max_conn() -> int:
    workers = _env_int("CELERY_WORKER_CONCURRENCY", 4, minimum=1)
    return max(16, workers * 4)


def _max_conn() -> int:
    return _env_int("POSTGRES_POOL_MAX_CONN", _default_max_conn(), minimum=1)


def pool_timeout_seconds() -> float:
    if os.getenv("POSTGRES_POOL_TIMEOUT") is not None:
        return _env_float("POSTGRES_POOL_TIMEOUT", _DEFAULT_POOL_TIMEOUT, minimum=0.0)
    return _env_float(
        "POSTGRES_POOL_GETCONN_TIMEOUT", _DEFAULT_POOL_TIMEOUT, minimum=0.0
    )


def _pool_is_closed(pool: object) -> bool:
    return bool(getattr(pool, "closed", False))


def _is_closed_pool_error(exc: BaseException) -> bool:
    return "closed" in str(exc).lower()


def _dbname_of(pool: object) -> Optional[str]:
    name = getattr(pool, "dbname", None)
    if isinstance(name, str) and name:
        return name
    kwargs = getattr(pool, "_kwargs", None)
    if isinstance(kwargs, dict):
        dbname = kwargs.get("dbname")
        if isinstance(dbname, str) and dbname:
            return dbname
    return None


def _connection_broken(conn: object) -> bool:
    try:
        return bool(getattr(conn, "closed", 1))
    except Exception:
        return True


def _reset_before_return(conn: object) -> bool:
    if _connection_broken(conn):
        return True
    try:
        status = getattr(conn, "status", TRANSACTION_STATUS_IDLE)
        if status != TRANSACTION_STATUS_IDLE:
            conn.rollback()
        return False
    except Exception:
        try:
            conn.close()
        except Exception:
            pass
        return True


def _close_quietly(conn: object) -> None:
    try:
        if conn is not None and not _connection_broken(conn):
            conn.close()
    except Exception:
        pass


def _putconn_quietly(
    pool: psycopg2.pool.ThreadedConnectionPool,
    conn: object,
    *,
    close: bool,
) -> None:
    if conn is None:
        return
    try:
        if _pool_is_closed(pool):
            _close_quietly(conn)
            return
        pool.putconn(conn, close=close)
    except Exception:
        _close_quietly(conn)


def _attach_wait_limiter(
    pool: psycopg2.pool.ThreadedConnectionPool, maxconn: int
) -> threading.BoundedSemaphore:
    limiter = getattr(pool, "_wait_limiter", None)
    if limiter is not None:
        return limiter
    limiter = threading.BoundedSemaphore(maxconn)
    pool.maxconn = maxconn
    pool._wait_limiter = limiter
    return limiter


def _wait_limiter_of(
    pool: psycopg2.pool.ThreadedConnectionPool,
) -> threading.BoundedSemaphore:
    maxconn = int(getattr(pool, "maxconn", None) or _max_conn())
    return _attach_wait_limiter(pool, maxconn)


class _PoolRegistry:
    def __init__(self) -> None:
        self._pools: "OrderedDict[str, psycopg2.pool.ThreadedConnectionPool]" = (
            OrderedDict()
        )
        self._pins: dict[str, int] = {}
        self._lock = threading.RLock()
        self._pid: Optional[int] = None

    def reset_after_fork(self) -> None:
        with self._lock:
            self._drop_inherited_pools()
            self._pid = os.getpid()

    def _drop_inherited_pools(self) -> None:
        self._pools.clear()
        self._pins.clear()

    def _reset_if_forked(self) -> None:
        current = os.getpid()
        if self._pid is not None and self._pid != current:
            self._drop_inherited_pools()
        self._pid = current

    def _create_pool(
        self,
        dbname: str,
        minconn: int | None = None,
        maxconn: int | None = None,
    ) -> psycopg2.pool.ThreadedConnectionPool:
        config.postgresql.validate_credentials()
        maxconn = _max_conn() if maxconn is None else max(1, int(maxconn))
        minconn = _min_conn() if minconn is None else max(0, int(minconn))
        minconn = min(minconn, maxconn)
        pool = psycopg2.pool.ThreadedConnectionPool(
            minconn=minconn,
            maxconn=maxconn,
            host=config.postgresql.host,
            port=config.postgresql.port,
            user=config.postgresql.username,
            password=config.postgresql.password,
            dbname=dbname,
        )
        pool.dbname = dbname
        pool.maxconn = maxconn
        pool._wait_limiter = threading.BoundedSemaphore(maxconn)
        return pool

    def _discard_locked(
        self,
        dbname: str,
        pool: Optional[psycopg2.pool.ThreadedConnectionPool],
        *,
        close: bool,
    ) -> None:
        current = self._pools.get(dbname)
        if current is pool:
            self._pools.pop(dbname, None)
        if close and pool is not None and not _pool_is_closed(pool):
            try:
                pool.closeall()
            except Exception as exc:
                logger.debug("Failed to close pool %s: %s", dbname, exc)

    def _live_pool(self, dbname: str) -> Optional[psycopg2.pool.ThreadedConnectionPool]:
        existing = self._pools.get(dbname)
        if existing is None:
            return None
        if not _pool_is_closed(existing):
            self._pools.move_to_end(dbname)
            return existing
        self._discard_locked(dbname, existing, close=False)
        return None

    def _insert_pool(
        self, dbname: str, pool: psycopg2.pool.ThreadedConnectionPool
    ) -> psycopg2.pool.ThreadedConnectionPool:
        self._pools[dbname] = pool
        self._evict_if_needed()
        return pool

    def _adopt_created(
        self, dbname: str, created: psycopg2.pool.ThreadedConnectionPool
    ) -> psycopg2.pool.ThreadedConnectionPool:
        self._reset_if_forked()
        existing = self._live_pool(dbname)
        if existing is not None:
            try:
                created.closeall()
            except Exception as exc:
                logger.debug("Dropped racing pool %s: %s", dbname, exc)
            return existing
        return self._insert_pool(dbname, created)

    def _ensure_pool(
        self,
        dbname: str,
        minconn: int | None = None,
        maxconn: int | None = None,
    ) -> psycopg2.pool.ThreadedConnectionPool:
        with self._lock:
            self._reset_if_forked()
            existing = self._live_pool(dbname)
            if existing is not None:
                return existing
        created = self._create_pool(dbname, minconn=minconn, maxconn=maxconn)
        with self._lock:
            return self._adopt_created(dbname, created)

    def get(
        self,
        dbname: str,
        minconn: int | None = None,
        maxconn: int | None = None,
        **_kwargs,
    ) -> psycopg2.pool.ThreadedConnectionPool:
        return self._ensure_pool(dbname, minconn=minconn, maxconn=maxconn)

    def _pin(self, dbname: str) -> None:
        self._pins[dbname] = self._pins.get(dbname, 0) + 1

    def _unpin(self, dbname: str) -> None:
        remaining = self._pins.get(dbname, 0) - 1
        if remaining > 0:
            self._pins[dbname] = remaining
        else:
            self._pins.pop(dbname, None)
        self._evict_if_needed()

    def _evict_if_needed(self) -> None:
        limit = _max_open_pools()
        while len(self._pools) > limit:
            victim = self._idle_lru_name()
            if victim is None:
                logger.warning(
                    "Postgres pool LRU is over cap (%s>%s) with all pools in use",
                    len(self._pools),
                    limit,
                )
                return
            evicted = self._pools.pop(victim)
            self._pins.pop(victim, None)
            try:
                evicted.closeall()
            except Exception as exc:
                logger.warning("Failed to close LRU pool %s: %s", victim, exc)
            else:
                logger.debug("Evicted LRU pool: %s", victim)

    def _idle_lru_name(self) -> Optional[str]:
        for name in self._pools:
            if self._pins.get(name, 0) <= 0:
                return name
        return None

    def close(self, dbname: str) -> None:
        with self._lock:
            pool = self._pools.get(dbname)
            self._discard_locked(dbname, pool, close=True)

    def _discard_closed(
        self, dbname: str, pool: psycopg2.pool.ThreadedConnectionPool
    ) -> None:
        with self._lock:
            self._discard_locked(dbname, pool, close=False)

    def _pin_if_current(
        self, dbname: str, pool: psycopg2.pool.ThreadedConnectionPool, pinned: bool
    ) -> bool:
        with self._lock:
            if self._pools.get(dbname) is not pool or _pool_is_closed(pool):
                return False
            if not pinned:
                self._pin(dbname)
            return True

    def acquire(self, dbname: str) -> object:
        timeout = pool_timeout_seconds()
        deadline = time.monotonic() + timeout
        pinned = False
        pool: Optional[psycopg2.pool.ThreadedConnectionPool] = None
        try:
            while True:
                pool = self._ensure_pool(dbname)
                if not self._pin_if_current(dbname, pool, pinned):
                    if pinned:
                        with self._lock:
                            self._unpin(dbname)
                        pinned = False
                    if time.monotonic() >= deadline:
                        raise PoolTimeout("connection pool timeout")
                    continue
                pinned = True
                remaining = deadline - time.monotonic()
                try:
                    return getconn_wait(pool, timeout=max(0.0, remaining))
                except PoolTimeout:
                    closed = _pool_is_closed(pool) or self._pools.get(dbname) is not pool
                    if not closed:
                        raise
                    self._discard_closed(dbname, pool)
                    with self._lock:
                        self._unpin(dbname)
                    pinned = False
                    if time.monotonic() >= deadline:
                        raise
        except Exception:
            if pinned:
                with self._lock:
                    self._unpin(dbname)
            raise

    def release(self, dbname: str, conn: object, *, close: bool) -> None:
        with self._lock:
            pool = self._pools.get(dbname)
        discard = close or _reset_before_return(conn)
        if pool is not None:
            _putconn_quietly(pool, conn, close=discard)
        else:
            _close_quietly(conn)
        limiter = getattr(pool, "_wait_limiter", None) if pool is not None else None
        if limiter is not None:
            limiter.release()
        with self._lock:
            self._unpin(dbname)


_registry = _PoolRegistry()


def reset_pools_after_fork() -> None:
    _registry.reset_after_fork()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=reset_pools_after_fork)


def get_pool(dbname: str, **kwargs) -> psycopg2.pool.ThreadedConnectionPool:
    return _registry.get(dbname, **kwargs)


def get_system_pool() -> psycopg2.pool.ThreadedConnectionPool:
    return _registry.get(_system_dbname())


def get_brain_pool(brain_id: str) -> psycopg2.pool.ThreadedConnectionPool:
    return _registry.get(brain_db_name(brain_id))


def _maintenance_dbname() -> str:
    return (
        getattr(config.postgresql, "maintenance_database", None)
        or _MAINTENANCE_FALLBACK_DB
    )


def _system_dbname() -> str:
    return getattr(config.postgresql, "system_database", None) or config.postgresql.database


def getconn_wait(
    pool: psycopg2.pool.ThreadedConnectionPool,
    timeout: float | None = None,
) -> psycopg2.extensions.connection:
    wait = pool_timeout_seconds() if timeout is None else max(0.0, float(timeout))
    limiter = _wait_limiter_of(pool)
    if not limiter.acquire(timeout=wait):
        raise PoolTimeout("connection pool timeout")
    try:
        return pool.getconn()
    except Exception as exc:
        limiter.release()
        if isinstance(exc, psycopg2.pool.PoolError):
            raise PoolTimeout("connection pool timeout") from exc
        raise


@contextmanager
def borrow(
    pool: psycopg2.pool.ThreadedConnectionPool,
) -> Iterator[psycopg2.extensions.connection]:
    dbname = _dbname_of(pool)
    close = False
    if dbname is None:
        conn = getconn_wait(pool)
        try:
            yield conn
        except (psycopg2.InterfaceError, psycopg2.OperationalError):
            close = True
            raise
        finally:
            discard = close or _reset_before_return(conn)
            _putconn_quietly(pool, conn, close=discard)
            limiter = getattr(pool, "_wait_limiter", None)
            if limiter is not None:
                limiter.release()
        return
    conn = _registry.acquire(dbname)
    try:
        yield conn
    except (psycopg2.InterfaceError, psycopg2.OperationalError):
        close = True
        raise
    finally:
        _registry.release(dbname, conn, close=close)


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
