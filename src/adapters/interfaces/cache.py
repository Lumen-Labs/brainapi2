"""
File: /cache.py
Created Date: Sunday October 19th 2025
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Sunday October 19th 2025 9:01:18 am
Modified By: the developer formerly known as Christian Nonis at <alch.infoemail@gmail.com>
-----
"""

from abc import ABC, abstractmethod


class CacheClient(ABC):
    """
    Abstract base class for cache clients.
    """

    @abstractmethod
    def get(self, key: str) -> str:
        """
        Get a value from the cache.
        """
        raise NotImplementedError("get method not implemented")

    @abstractmethod
    def set(self, key: str, value: str, expires_in: int) -> bool:
        """
        Set a value in the cache with an expiration time.

        Args:
            key: The key to set.
            value: The value to set.
            expires_in: The expiration time in seconds.

        Returns:
            True if the value was set, False otherwise.
        """
        raise NotImplementedError("set method not implemented")

    @abstractmethod
    def delete(self, key: str) -> bool:
        """
        Delete a value from the cache.
        """
        raise NotImplementedError("delete method not implemented")
