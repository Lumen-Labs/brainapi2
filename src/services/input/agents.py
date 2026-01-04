"""
File: /agents.py
Created Date: Sunday December 21st 2025
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Sunday December 21st 2025 4:17:39 pm
Modified By: the developer formerly known as Christian Nonis at <alch.infoemail@gmail.com>
-----
"""

from src.adapters.cache import CacheAdapter
from src.adapters.embeddings import EmbeddingsAdapter
from src.adapters.graph import GraphAdapter
from src.adapters.llm import LLMAdapter

from src.adapters.embeddings import VectorStoreAdapter
from src.core.agents.architect_agent import ArchitectAgent
from src.core.agents.janitor_agent import JanitorAgent
from src.core.agents.scout_agent import ScoutAgent

from src.lib.llm.client_small import _llm_small_client
from src.lib.redis.client import _redis_client
from src.lib.neo4j.client import _neo4j_client
from src.lib.embeddings.client import _embeddings_client
from src.lib.milvus.client import _milvus_client


# Initialze the adapters ================================
llm_small_adapter = LLMAdapter()
llm_small_adapter.add_client(_llm_small_client)

cache_adapter = CacheAdapter()
cache_adapter.add_client(_redis_client)

graph_adapter = GraphAdapter()
graph_adapter.add_client(_neo4j_client)

embeddings_adapter = EmbeddingsAdapter()
embeddings_adapter.add_client(_embeddings_client)

vector_store_adapter = VectorStoreAdapter()
vector_store_adapter.add_client(_milvus_client)


# Initialize the agents ================================
scout_agent = ScoutAgent(
    llm_small_adapter,
    cache_adapter,
    kg=graph_adapter,
    vector_store=vector_store_adapter,
    embeddings=embeddings_adapter,
    # database_desc=_neo4j_client.graphdb_description,
)
architect_agent = ArchitectAgent(
    llm_small_adapter,
    cache_adapter,
    kg=graph_adapter,
    vector_store=vector_store_adapter,
    embeddings=embeddings_adapter,
    # database_desc=_neo4j_client.graphdb_description,
)
janitor_agent = JanitorAgent(
    llm_small_adapter,
    kg=graph_adapter,
    vector_store=vector_store_adapter,
    embeddings=embeddings_adapter,
    database_desc=_neo4j_client.graphdb_description,
)
