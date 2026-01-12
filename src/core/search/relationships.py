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
    brain_id: str = "default",
) -> SearchRelationshipsResult:
    """
    Search for relationships in the knowledge graph using optional filters and full-text query.
    
    Parameters:
        limit (int): Maximum number of relationship records to return.
        skip (int): Number of matching records to skip (offset).
        relationship_types (list[str] | None): List of relationship type names to include; when None, all types are allowed.
        from_node_labels (list[str] | None): List of labels to filter source nodes; when None, any source node label is allowed.
        to_node_labels (list[str] | None): List of labels to filter target nodes; when None, any target node label is allowed.
        query_text (str | None): Full-text query string applied to the specified search target; when None, no full-text filtering is applied.
        query_search_target (str | None): Field to apply the full-text query to (e.g., "all", "from", "to", "relationship"); defaults to "all".
        brain_id (str): Identifier of the brain/graph context to search within.
    
    Returns:
        SearchRelationshipsResult: Search results containing matching relationships and associated metadata.
    """
    result = graph_adapter.search_relationships(
        brain_id,
        limit,
        skip,
        relationship_types,
        from_node_labels,
        to_node_labels,
        query_text,
        query_search_target,
    )
    return result