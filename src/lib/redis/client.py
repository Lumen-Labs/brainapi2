"""
File: /client.py
Created Date: Sunday October 19th 2025
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Sunday October 19th 2025 12:37:27 pm
Modified By: the developer formerly known as Christian Nonis at <alch.infoemail@gmail.com>
-----
"""

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

    def get(self, key: str) -> str:
        """
        Get a value from the cache.
        """
        return self.client.get(key)

    def set(self, key: str, value: str, expires_in: int) -> bool:
        """
        Set a value in the cache with an expiration time.
        """
        return self.client.set(key, value, ex=expires_in)

    def delete(self, key: str) -> bool:
        """
        Delete a value from the cache.
        """
        return self.client.delete(key)


_redis_client = RedisClient()
