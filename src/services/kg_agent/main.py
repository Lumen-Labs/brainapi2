"""
File: /main.py
Created Date: Sunday October 19th 2025
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Sunday October 19th 2025 10:28:43 am
Modified By: the developer formerly known as Christian Nonis at <alch.infoemail@gmail.com>
-----
"""

from src.adapters.embeddings import EmbeddingsAdapter, VectorStoreAdapter
from src.adapters.graph import GraphAdapter
from src.adapters.llm import LLMAdapter
from src.core.agents.kg_agent import KGAgent

from src.lib.llm.client_large import _llm_large_client
from src.lib.redis.client import _redis_client
from src.adapters.cache import CacheAdapter
from src.lib.neo4j.client import _neo4j_client
from src.lib.embeddings.client import _embeddings_client
from src.lib.milvus.client import _milvus_client

# Initialze the adapters
llm_large_adapter = LLMAdapter()
llm_large_adapter.add_client(_llm_large_client)

cache_adapter = CacheAdapter()
cache_adapter.add_client(_redis_client)

graph_adapter = GraphAdapter()
graph_adapter.add_client(_neo4j_client)

embeddings_adapter = EmbeddingsAdapter()
embeddings_adapter.add_client(_embeddings_client)

vector_store_adapter = VectorStoreAdapter()
vector_store_adapter.add_client(_milvus_client)

# Initialize the knowledge graph agent
kg_agent = KGAgent(
    llm_large_adapter,
    cache_adapter,
    kg=graph_adapter,
    vector_store=vector_store_adapter,
    embeddings=embeddings_adapter,
    database_desc=_neo4j_client.graphdb_description,
)
