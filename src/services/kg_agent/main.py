"""
File: /main.py
Created Date: Sunday October 19th 2025
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Sunday October 19th 2025 10:28:43 am
Modified By: the developer formerly known as Christian Nonis at <alch.infoemail@gmail.com>
-----
"""

from src.adapters.cache import CacheAdapter
from src.adapters.embeddings import EmbeddingsAdapter, VectorStoreAdapter
from src.adapters.graph import GraphAdapter
from src.adapters.llm import LLMAdapter
from src.config import config
from src.core.agents.kg_agent import KGAgent
from src.lib.milvus.client import _milvus_client
from src.lib.neo4j.client import _neo4j_client
from src.lib.redis.client import _redis_client

_MODES = ("local", "remote")


def _require_mode():
    if config.models_mode not in _MODES:
        raise ValueError(f"Invalid models mode: {config.models_mode}")
    return config.models_mode == "local"


_is_local = _require_mode()

if _is_local:
    from src.lib.llm.client_ollama import _llm_large_client

    _llm_large = LLMAdapter()
    _llm_large.add_client(_llm_large_client)
else:
    from src.lib.llm.client_large import _llm_large_client

    _llm_large = LLMAdapter()
    _llm_large.add_client(_llm_large_client)

llm_large_adapter = _llm_large

cache_adapter = CacheAdapter()
cache_adapter.add_client(_redis_client)

graph_adapter = GraphAdapter()
graph_adapter.add_client(_neo4j_client)

if _is_local:
    from src.lib.embeddings.local import _embeddings_local_client

    _embeddings = EmbeddingsAdapter()
    _embeddings.add_client(_embeddings_local_client)
else:
    from src.lib.embeddings.client import _embeddings_client

    _embeddings = EmbeddingsAdapter()
    _embeddings.add_client(_embeddings_client)

embeddings_adapter = _embeddings

vector_store_adapter = VectorStoreAdapter()
vector_store_adapter.add_client(_milvus_client)

kg_agent = KGAgent(
    llm_large_adapter,
    cache_adapter,
    kg=graph_adapter,
    vector_store=vector_store_adapter,
    embeddings=embeddings_adapter,
    database_desc=graph_adapter.graphdb_description,
)
