"""
File: /relationships.py
Created Date: Friday November 7th 2025
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Friday November 7th 2025 8:12:57 pm
Modified By: the developer formerly known as Christian Nonis at <alch.infoemail@gmail.com>
-----
"""

from src.constants.kg import Triple
from src.services.kg_agent.main import graph_adapter


def search_relationships(limit: int = 10, skip: int = 0) -> list[Triple]:
    """
    Search the relationships of the graph.
    """
    result = graph_adapter.search_relationships(limit, skip)
    return result
