"""
File: /kg.py
Created Date: Wednesday December 31st 2025
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Wednesday December 31st 2025 7:45:55 pm
Modified By: Christian Nonis <alch.infoemail@gmail.com>
-----
"""

import asyncio
from src.services.kg_agent.main import (
    embeddings_adapter,
    graph_adapter,
    vector_store_adapter,
)


async def get_hops(
    query: str, degrees: int = 2, flattened: bool = True, brain_id: str = "default"
):
    """
    Get the hops in the graph for a given query.

    Currently only supports 2nd degree hops.
    """
    text_embeddings = embeddings_adapter.embed_text(query)
    data_vectors = vector_store_adapter.search_vectors(
        text_embeddings.embeddings, store="nodes", brain_id=brain_id
    )
    node_uuids = [v.metadata.get("uuid") for v in data_vectors]
    print("[+] Getting 2nd degree hops for nodes:", node_uuids)
    if len(node_uuids) == 0:
        return []

    hops = await asyncio.to_thread(
        graph_adapter.get_2nd_degree_hops,
        node_uuids,
        flattened=flattened,
        vector_store_adapter=vector_store_adapter,
        brain_id=brain_id,
        similarity_threshold=0.0,
    )
    return hops
