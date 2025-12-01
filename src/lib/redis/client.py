"""
File: /client.py
Created Date: Sunday October 19th 2025
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Sunday October 19th 2025 12:37:27 pm
Modified By: the developer formerly known as Christian Nonis at <alch.infoemail@gmail.com>
-----
"""

from typing import Optional
from redis import Redis
from redis.connection import ConnectionPool
from src.adapters.interfaces.cache import CacheClient
from src.config import config


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
        return self.client.get(prefixed_key)

    def set(
        self, key: str, value: str, brain_id: str, expires_in: Optional[int] = None
    ) -> bool:
        """
        Set a value in the cache with an expiration time.
        """
        prefixed_key = self._get_key(key, brain_id)
        return self.client.set(
            prefixed_key, value, **({"ex": expires_in} if expires_in else {})
        )

    def delete(self, key: str, brain_id: str) -> bool:
        """
        Delete a value from the cache.
        """
        prefixed_key = self._get_key(key, brain_id)
        return self.client.delete(prefixed_key)


_redis_client = RedisClient()
