from src.core.agents.kg_agent import KGAgent
from src.core.instances import (
    cache_adapter,
    embeddings_adapter,
    graph_adapter,
    llm_large_adapter,
    vector_store_adapter,
)

kg_agent = KGAgent(
    llm_large_adapter,
    cache_adapter,
    kg=graph_adapter,
    vector_store=vector_store_adapter,
    embeddings=embeddings_adapter,
    database_desc=graph_adapter.graphdb_description,
)

__all__ = [
    "cache_adapter",
    "embeddings_adapter",
    "graph_adapter",
    "kg_agent",
    "llm_large_adapter",
    "vector_store_adapter",
]
