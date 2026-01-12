"""
File: /entities.py
Created Date: Friday November 7th 2025
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Friday November 7th 2025 8:13:02 pm
Modified By: the developer formerly known as Christian Nonis at <alch.infoemail@gmail.com>
-----
"""

from typing import Optional
from src.constants.kg import SearchEntitiesResult
from src.services.kg_agent.main import graph_adapter


def search_entities(
    limit: int = 10,
    skip: int = 0,
    node_labels: Optional[list[str]] = None,
    query_text: Optional[str] = None,
    brain_id: str = "default",
) -> SearchEntitiesResult:
    """
    Search entities in the knowledge graph.
    
    Parameters:
        limit (int): Maximum number of entities to return.
        skip (int): Number of entities to skip (offset) for pagination.
        node_labels (Optional[list[str]]): Filter results to nodes matching any of these labels.
        query_text (Optional[str]): Text to match against entity properties or content.
        brain_id (str): Identifier of the graph/brain to query.
    
    Returns:
        SearchEntitiesResult: Result object containing matched entities and pagination metadata.
    """
    result = graph_adapter.search_entities(
        brain_id, limit, skip, node_labels, query_text
    )
    return result