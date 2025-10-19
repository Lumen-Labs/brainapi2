"""
File: /graph.py
Created Date: Sunday October 19th 2025
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Sunday October 19th 2025 9:01:05 am
Modified By: the developer formerly known as Christian Nonis at <alch.infoemail@gmail.com>
-----
"""

from abc import ABC, abstractmethod


class GraphClient(ABC):
    """
    Abstract base class for graph clients.
    """

    @property
    @abstractmethod
    def graphdb_type(self) -> str:
        """
        Get the type of graph database.
        """
        raise NotImplementedError("graphdb_type method not implemented")

    @abstractmethod
    def execute_operation(self, operation: str) -> str:
        """
        Execute a graph operation.
        """
        raise NotImplementedError("execute_operation method not implemented")
