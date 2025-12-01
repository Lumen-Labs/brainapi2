"""
File: /cache.py
Created Date: Sunday October 19th 2025
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Sunday October 19th 2025 9:00:36 am
Modified By: the developer formerly known as Christian Nonis at <alch.infoemail@gmail.com>
-----
"""

from typing import Optional
from .interfaces.cache import CacheClient


class CacheAdapter:
    """
    Adapter for the cache client.
    """

    def __init__(self):
        self.cache = None

    def add_client(self, client: CacheClient) -> None:
        """
        Add a cache client to the adapter.
        """
        self.cache = client

    def get(self, key: str, brain_id: str = "default") -> str:
        """
        Get a value from the cache.
        """
        return self.cache.get(key, brain_id)

    def set(
        self,
        key: str,
        value: str,
        brain_id: str = "default",
        expires_in: Optional[int] = None,
    ) -> bool:
        """
        Set a value in the cache with an expiration time.
        """
        return self.cache.set(key, value, brain_id, expires_in)

    def delete(self, key: str, brain_id: str = "default") -> bool:
        """
        Delete a value from the cache.
        """
        return self.cache.delete(key, brain_id)
