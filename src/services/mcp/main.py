"""
File: /main.py
Project: mcp
Created Date: Tuesday February 10th 2026
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Tuesday February 10th 2026 8:30:11 pm
Modified By: Christian Nonis <alch.infoemail@gmail.com>
-----
"""

from typing import Any
from mcp.server import FastMCP
from pydantic import BaseModel
from src.lib.neo4j.client import _neo4j_client
from src.services.data.main import data_adapter
from src.services.kg_agent.main import graph_adapter

mcp = FastMCP("brainapi-mcp", stateless_http=True)


@mcp.tool()
def get_search_operation_instructions(message: str) -> str:
    """
    This tool will provide instructions on how to use the search_memory tool.
    """
    return f"""
    The brains are a storage for information and memories, they are powered by multiple dbs.
    The `search_memory` tool will execute graph operations to search the knowledge graph that contains informations.
    {_neo4j_client.graphdb_description}.
    The input must be a JSON object with the following fields:
    - db_query: str: the operation to execute on the graph.
    - brain_id: str: the brain to search in.
    """


class SearchMemoryInput(BaseModel):
    db_query: str
    brain_id: str


@mcp.tool()
def search_memory(db_query: str, brain_id: str) -> Any:
    """
    Search the brain for memories and information.
    This tool will search into the knowledge graph.

    Input must be a JSON object with the following fields:
    - db_query: str: the operation to execute on the graph.
    - brain_id: str: the brain to search in.
    """
    try:
        return graph_adapter.execute_operation(db_query, brain_id)
    except Exception as e:
        return f"Error executing graph operation: {e}"


@mcp.tool()
def list_brains() -> list[str]:
    """
    This tool lists all the brains/memory stores available
    """
    brains = data_adapter.get_brains_list()
    return [brain.name_key for brain in brains]
