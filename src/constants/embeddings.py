"""
File: /embeddings.py
Created Date: Friday October 24th 2025
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Friday October 24th 2025 7:56:49 pm
Modified By: the developer formerly known as Christian Nonis at <alch.infoemail@gmail.com>
-----
"""

from typing import Optional
from pydantic import BaseModel, Field

EMBEDDING_NODES_DIMENSION = 3072
EMBEDDING_TRIPLETS_DIMENSION = 3072
OBSERVATIONS_DIMENSION = 3072
DATA_DIMENSION = 3072

EMBEDDING_STORES_SIZES = {
    "nodes": EMBEDDING_NODES_DIMENSION,
    "triplets": EMBEDDING_TRIPLETS_DIMENSION,
    "observations": OBSERVATIONS_DIMENSION,
    "data": DATA_DIMENSION,
}


class Vector(BaseModel):
    """
    Embedding node model.
    """

    id: str
    embeddings: Optional[list[float]]
    metadata: dict

    distance: Optional[float] = Field(
        default=None,
        description=(
            "The distance between the query and the result. "
            "This is only available for search results."
        ),
    )
