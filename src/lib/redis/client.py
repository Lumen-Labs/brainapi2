"""
File: /client.py
Created Date: Sunday October 19th 2025
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Sunday October 19th 2025 12:37:27 pm
Modified By: the developer formerly known as Christian Nonis at <alch.infoemail@gmail.com>
-----
"""

import json
import time
from typing import Optional
from redis import Redis
from redis.connection import ConnectionPool
from src.adapters.interfaces.cache import CacheClient
from src.config import config

TASK_RETENTION_SECONDS = 3600 * 24 * 7


class RedisClient(CacheClient):
    """
    Redis client.
    """

    def __init__(self):
        pool = ConnectionPool(
            host=config.redis.host,
            port=config.redis.port,
            socket_connect_timeout=5,
            socket_timeout=5,
            retry_on_timeout=True,
            health_check_interval=30,
            max_connections=50,
        )
        self.client = Redis(connection_pool=pool)

    def _get_key(self, key: str, brain_id: str) -> str:
        """
        Get the prefixed key for a given brain_id.
        """
        return f"{brain_id}:{key}"

    def get(self, key: str, brain_id: str) -> str:
        """
        Get a value from the cache.
        """
        prefixed_key = self._get_key(key, brain_id)
        result = self.client.get(prefixed_key)
        if result is None:
            return None
        if isinstance(result, bytes):
            return result.decode("utf-8")
        return result

    def set(
        self, key: str, value: str, brain_id: str, expires_in: Optional[int] = None
    ) -> bool:
        """
        Set a value in the cache with an expiration time.
        Task keys are stored WITHOUT TTL to prevent eviction by volatile-lru.
        """
        prefixed_key = self._get_key(key, brain_id)
        if key.startswith("task:"):
            task_id = key.split(":")[-1]
            self.client.hset(
                f"{brain_id}:_tasks",
                task_id,
                json.dumps({"data": value, "created_at": time.time()}),
            )
            return True
        return self.client.set(
            prefixed_key, value, **({"ex": expires_in} if expires_in else {})
        )

    def delete(self, key: str, brain_id: str) -> bool:
        """
        Delete a value from the cache.
        """
        prefixed_key = self._get_key(key, brain_id)
        if key.startswith("task:"):
            task_id = key.split(":")[-1]
            return self.client.hdel(f"{brain_id}:_tasks", task_id)
        return self.client.delete(prefixed_key)

    def get_task_keys(self, brain_id: str) -> list[str]:
        """
        Get all task keys for a given brain_id.
        Also cleans up expired tasks (older than TASK_RETENTION_SECONDS).
        """
        tasks_hash = f"{brain_id}:_tasks"
        all_tasks = self.client.hgetall(tasks_hash)
        valid_keys = []
        now = time.time()
        expired_keys = []

        for task_id, task_data in all_tasks.items():
            task_id_str = (
                task_id.decode("utf-8") if isinstance(task_id, bytes) else task_id
            )
            try:
                data = json.loads(task_data)
                created_at = data.get("created_at", 0)
                if now - created_at > TASK_RETENTION_SECONDS:
                    expired_keys.append(task_id_str)
                else:
                    valid_keys.append(f"task:{task_id_str}")
            except (json.JSONDecodeError, TypeError):
                expired_keys.append(task_id_str)

        if expired_keys:
            self.client.hdel(tasks_hash, *expired_keys)

        return valid_keys

    def get_task(self, task_id: str, brain_id: str) -> Optional[str]:
        """
        Get a specific task by ID.
        """
        task_data = self.client.hget(f"{brain_id}:_tasks", task_id)
        if task_data is None:
            return None
        try:
            data = json.loads(task_data)
            return data.get("data")
        except (json.JSONDecodeError, TypeError):
            return None


_redis_client = RedisClient()
