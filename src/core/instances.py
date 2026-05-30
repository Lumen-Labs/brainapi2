from src.adapters.cache import CacheAdapter
from src.adapters.data import DataAdapter
from src.adapters.embeddings import EmbeddingsAdapter, VectorStoreAdapter
from src.adapters.graph import GraphAdapter
from src.adapters.llm import LLMAdapter
from src.config import config
from src.lib.redis.client import _redis_client

def _build_small_llm(provider: str):
    if provider == "ollama":
        from src.lib.llm.client_ollama import _llm_small_client

        return _llm_small_client
    if provider == "azure":
        from src.lib.llm.client_azure import _llm_small_client_azure

        return _llm_small_client_azure
    if provider == "openai":
        from src.lib.llm.client_openai import _llm_small_client_openai

        return _llm_small_client_openai
    if provider == "gcp_vertex":
        from src.lib.llm.client_small import _llm_small_client

        return _llm_small_client
    if provider == "amazon_bedrock":
        from src.lib.llm.client_bedrock import _llm_small_client_bedrock

        return _llm_small_client_bedrock
    raise ValueError(f"Unsupported small LLM provider: {provider}")


def _build_large_llm(provider: str):
    if provider == "ollama":
        from src.lib.llm.client_ollama import _llm_large_client

        return _llm_large_client
    if provider == "azure":
        from src.lib.llm.client_large import _llm_large_client

        return _llm_large_client
    if provider == "openai":
        from src.lib.llm.client_openai import _llm_large_client_openai

        return _llm_large_client_openai
    if provider == "gcp_vertex":
        from src.lib.llm.client_vertex import _llm_large_client_vertex

        return _llm_large_client_vertex
    if provider == "amazon_bedrock":
        from src.lib.llm.client_bedrock import _llm_large_client_bedrock

        return _llm_large_client_bedrock
    raise ValueError(f"Unsupported large LLM provider: {provider}")


def _build_embeddings(provider: str):
    if provider == "ollama":
        from src.lib.embeddings.client_ollama import _embeddings_ollama_client

        return _embeddings_ollama_client
    if provider == "azure":
        from src.lib.embeddings.client import _embeddings_client

        return _embeddings_client
    if provider == "openai":
        from src.lib.embeddings.client_openai import _embeddings_openai_client

        return _embeddings_openai_client
    if provider == "gcp_vertex":
        from src.lib.embeddings.client_vertex import _embeddings_vertex_client

        return _embeddings_vertex_client
    if provider == "amazon_bedrock":
        from src.lib.embeddings.client_bedrock import _embeddings_bedrock_client

        return _embeddings_bedrock_client
    raise ValueError(f"Unsupported embeddings provider: {provider}")


_llm_small = LLMAdapter()
_llm_small.add_client(_build_small_llm(config.llm_small_provider))
_llm_large = LLMAdapter()
_llm_large.add_client(_build_large_llm(config.llm_large_provider))

llm_small_adapter = _llm_small
llm_large_adapter = _llm_large

cache_adapter = CacheAdapter()
cache_adapter.add_client(_redis_client)

graph_adapter = GraphAdapter()
if config.graph_db == "neo4j":
    from src.lib.neo4j.client import _neo4j_client

    graph_adapter.add_client(_neo4j_client)
elif config.graph_db == "networkx":
    from src.lib.postgresql.networkx import get_networkx_graph_client

    graph_adapter.add_client(get_networkx_graph_client())

if config.data_db == "mongo":
    from src.lib.mongo.client import _mongo_client

    data_adapter = DataAdapter()
    data_adapter.add_client(_mongo_client)
elif config.data_db == "postgresql":
    from src.lib.postgresql.data import get_postgresql_data_client

    data_adapter = DataAdapter()
    data_adapter.add_client(get_postgresql_data_client())

from src.lib.embeddings.client_small import _embeddings_small_client

_embeddings_primary = EmbeddingsAdapter()
_embeddings_primary.add_client(_build_embeddings(config.embeddings_provider))
_embeddings_small = EmbeddingsAdapter()
_embeddings_small.add_client(_embeddings_small_client)

embeddings_adapter = _embeddings_primary
embeddings_small_adapter = _embeddings_small

if config.vector_db == "milvus":
    from src.lib.milvus.client import _milvus_client

    vector_store_adapter = VectorStoreAdapter()
    vector_store_adapter.add_client(_milvus_client)
elif config.vector_db == "postgresql":
    from src.lib.postgresql.vectors import get_postgresql_vector_store_client

    vector_store_adapter = VectorStoreAdapter()
    vector_store_adapter.add_client(get_postgresql_vector_store_client())
