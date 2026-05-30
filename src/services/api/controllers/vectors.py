"""
File: /vectors.py
Created Date: Saturday May 30th 2026
Author: Christian Nonis <alch.infoemail@gmail.com>
"""

import asyncio

from fastapi.responses import JSONResponse

from src.constants.embeddings import EMBEDDING_STORES_SIZES
from src.services.kg_agent.main import vector_store_adapter


async def get_vector_stores():
    return {
        "stores": [
            {"name": name, "dimension": dim}
            for name, dim in EMBEDDING_STORES_SIZES.items()
        ]
    }


async def list_vectors(
    store: str,
    brain_id: str,
    limit: int = 10,
    skip: int = 0,
    include_embeddings: bool = False,
):
    vectors, total = await asyncio.to_thread(
        vector_store_adapter.list_vectors,
        store,
        brain_id,
        limit,
        skip,
        include_embeddings,
    )
    return JSONResponse(
        content={
            "message": "Vectors retrieved successfully",
            "store": store,
            "vectors": [v.model_dump(mode="json") for v in vectors],
            "total": total,
        }
    )
