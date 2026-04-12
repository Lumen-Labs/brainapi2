"""
File: /main.py
Project: mcp
Created Date: Tuesday February 10th 2026
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Sunday February 22nd 2026 5:21:48 pm
Modified By: Christian Nonis <alch.infoemail@gmail.com>
-----
"""

import asyncio
from contextvars import ContextVar
from typing import Any

from mcp.server import FastMCP

from src.core.instances import (
    data_adapter,
    embeddings_adapter,
    graph_adapter,
    vector_store_adapter,
)
from src.lib.neo4j.client import _neo4j_client
from src.services.mcp.utils import guard_brainpat

auth_token_var: ContextVar[str | None] = ContextVar("auth_token", default=None)

mcp = FastMCP("brainapi-mcp", stateless_http=True, host="0.0.0.0")


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


def _search_memory_sync(db_query: str, brain_id: str) -> Any:
    try:
        if not guard_brainpat(auth_token_var.get(), brain_id):
            return "Unauthorized"
        return graph_adapter.execute_operation(db_query, brain_id)
    except Exception as e:
        return f"Error executing graph operation: {e}"


@mcp.tool()
async def search_memory(db_query: str, brain_id: str) -> Any:
    """
    Search the brain for memories and information.
    This tool will search into the knowledge graph.

    Input must be a JSON object with the following fields:
    - db_query: str: the operation to execute on the graph.
    - brain_id: str: the brain to search in.
    """
    return await asyncio.to_thread(_search_memory_sync, db_query, brain_id)


def _search_semantically_sync(query: str, brain_id: str) -> Any:
    try:
        if not guard_brainpat(auth_token_var.get(), brain_id):
            return "Unauthorized"
        query_embedding = embeddings_adapter.embed_text(query)
        data_vectors = vector_store_adapter.search_vectors(
            query_embedding.embeddings, store="nodes", brain_id=brain_id, k=5
        )
        triplets = graph_adapter.get_event_centric_neighbors(
            [v.metadata.get("uuid") for v in data_vectors], brain_id=brain_id
        )
        return triplets
    except Exception as e:
        return f"Error executing graph operation: {e}"


@mcp.tool()
async def search_semantically(query: str, brain_id: str) -> Any:
    """
    Search information semantically, given a query and a brain to search in,
    this tool will return a list of graph nodes that are semantically related to the query.

    This tool is useful to search into the graph for information without knowing names or labels.

    Input must be a JSON object with the following fields:
    - query: str: the query to search for.
    - brain_id: str: the brain to search in.
    """
    return await asyncio.to_thread(_search_semantically_sync, query, brain_id)


def _list_brains_sync() -> list[str] | str:
    brain_key = guard_brainpat(auth_token_var.get())
    if not brain_key:
        return "Unauthorized"
    if type(brain_key) == str:
        return [brain_key]
    brains = data_adapter.get_brains_list()
    return [brain.name_key for brain in brains]


@mcp.tool()
async def list_brains() -> list[str] | str:
    """
    This tool lists all the brains/memory stores available
    """
    return await asyncio.to_thread(_list_brains_sync)
