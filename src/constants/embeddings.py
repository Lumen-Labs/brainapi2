"""
File: /embeddings.py
Created Date: Friday October 24th 2025
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Friday October 24th 2025 7:56:49 pm
Modified By: Christian Nonis <alch.infoemail@gmail.com>
-----
"""

from typing import Optional
from pydantic import BaseModel, Field
from src.config import config

EMBEDDING_NODES_DIMENSION = config.embeddings.embedding_nodes_dimension
EMBEDDING_TRIPLETS_DIMENSION = config.embeddings.embedding_triplets_dimension
OBSERVATIONS_DIMENSION = config.embeddings.embedding_observations_dimension
DATA_DIMENSION = config.embeddings.embedding_data_dimension
EMBEDDING_RELATIONSHIPS_DIMENSION = config.embeddings.embedding_relationships_dimension

EMBEDDING_STORES_SIZES = {
    "nodes": EMBEDDING_NODES_DIMENSION,
    "triplets": EMBEDDING_TRIPLETS_DIMENSION,
    "observations": OBSERVATIONS_DIMENSION,
    "data": DATA_DIMENSION,
    "relationships": EMBEDDING_RELATIONSHIPS_DIMENSION,
}


class Vector(BaseModel):
    """
    Embedding node model.
    """

    id: str
    embeddings: Optional[list[float]] = None
    metadata: dict

    distance: Optional[float] = Field(
        default=None,
        description=(
            "The distance between the query and the result. "
            "This is only available for search results."
        ),
    )
