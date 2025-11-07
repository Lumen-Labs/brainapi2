"""
File: /entities.py
Created Date: Friday November 7th 2025
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Friday November 7th 2025 8:13:02 pm
Modified By: the developer formerly known as Christian Nonis at <alch.infoemail@gmail.com>
-----
"""

from src.constants.kg import Node
from src.services.kg_agent.main import graph_adapter


def search_entities(limit: int = 10, skip: int = 0) -> list[Node]:
    """
    Search the entities of the graph.
    """
    result = graph_adapter.search_entities(limit, skip)
    return result
