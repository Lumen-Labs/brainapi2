import threading
import time

import psycopg2.pool
import pytest

from src.lib.postgresql._provisioning import PoolTimeout, borrow, getconn_wait


class FakePool:
    def __init__(self, maxconn=1):
        self.maxconn = maxconn
        self._wait_limiter = threading.BoundedSemaphore(maxconn)
        self.checked_out = 0
        self.lock = threading.Lock()

    def getconn(self):
        with self.lock:
            self.checked_out += 1
        return object()

    def putconn(self, _conn):
        with self.lock:
            self.checked_out -= 1


def test_pool_waits_then_acquires_when_a_slot_frees():
    pool = FakePool(maxconn=1)
    released = threading.Event()
    acquired = threading.Event()
    result = []

    def hold():
        with borrow(pool):
            acquired.set()
            released.wait(2)

    def wait_for_slot():
        conn = getconn_wait(pool, timeout=1)
        result.append(conn)
        pool.putconn(conn)
        pool._wait_limiter.release()

    holder = threading.Thread(target=hold)
    holder.start()
    assert acquired.wait(1)
    waiter = threading.Thread(target=wait_for_slot)
    waiter.start()
    time.sleep(0.05)
    assert not result
    released.set()
    holder.join(1)
    waiter.join(1)
    assert result


def test_pool_timeout_does_not_raise_pool_exhausted():
    pool = FakePool(maxconn=1)
    with borrow(pool):
        with pytest.raises(PoolTimeout):
            getconn_wait(pool, timeout=0)


def test_pool_error_after_acquire_becomes_timeout():
    pool = FakePool(maxconn=1)

    def fail():
        raise psycopg2.pool.PoolError("connection pool exhausted")

    pool.getconn = fail
    with pytest.raises(PoolTimeout):
        getconn_wait(pool, timeout=1)


def test_concurrent_waiters_never_exceed_maxconn():
    pool = FakePool(maxconn=2)
    current = 0
    peak = 0
    lock = threading.Lock()
    errors = []

    def worker():
        nonlocal current, peak
        try:
            with borrow(pool):
                with lock:
                    current += 1
                    peak = max(peak, current)
                time.sleep(0.05)
                with lock:
                    current -= 1
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(2)
    assert errors == []
    assert peak <= 2
