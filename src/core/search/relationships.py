"""
File: /relationships.py
Created Date: Friday November 7th 2025
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Friday November 7th 2025 8:12:57 pm
Modified By: the developer formerly known as Christian Nonis at <alch.infoemail@gmail.com>
-----
"""

from typing import Optional
from src.constants.kg import SearchRelationshipsResult
from src.services.kg_agent.main import graph_adapter


def search_relationships(
    limit: int = 10,
    skip: int = 0,
    relationship_types: Optional[list[str]] = None,
    from_node_labels: Optional[list[str]] = None,
    to_node_labels: Optional[list[str]] = None,
    query_text: Optional[str] = None,
    query_search_target: Optional[str] = "all",
) -> SearchRelationshipsResult:
    """
    Search the relationships of the graph.
    """
    result = graph_adapter.search_relationships(
        limit,
        skip,
        relationship_types,
        from_node_labels,
        to_node_labels,
        query_text,
        query_search_target,
    )
    return result
