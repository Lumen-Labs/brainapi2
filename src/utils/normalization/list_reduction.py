"""
File: /list_reduction.py
Created Date: Wednesday December 31st 2025
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Wednesday December 31st 2025 9:44:49 am
Modified By: Christian Nonis <alch.infoemail@gmail.com>
-----
"""

from typing import List, Optional, TypeVar
from src.utils.similarity.vectors import cosine_similarity
from pydantic import BaseModel

from src.lib.embeddings.client_small import EmbeddingsClientSmall

T = TypeVar("T")


def _get_nested_value(obj, key_path: str):
    """
    Get a value from an object using a dot-separated key path.
    Supports both object attributes and dictionary keys.

    Example: _get_nested_value(obj, "a.b.c") returns obj.a.b.c or obj["a"]["b"]["c"]
    """
    keys = key_path.split(".")
    current = obj

    for key in keys:
        if isinstance(current, dict):
            if key not in current:
                return None
            current = current[key]
        else:
            if not hasattr(current, key):
                return None
            current = getattr(current, key)

    return current


def _is_vector(value) -> bool:
    """
    Check if a value is a vector (list of numbers).
    """
    if not isinstance(value, (list, tuple)):
        return False
    if len(value) == 0:
        return False
    return all(isinstance(x, (int, float)) for x in value)


class Rerank(BaseModel):
    local: str
    with_: str


def reduce_list(
    list_: List[T],
    access_key: Optional[str] = None,
    similarity_threshold: float = 0.92,
    by_vector: Optional[list[float]] = None,
    rerank: Optional[Rerank] = None,
) -> List[T]:
    """
    Reduce a list of items by grouping similar items together.
    Returns a list with one representative item from each similarity group.

    If access_key points to a vector (list of floats), it will be used directly.
    If access_key points to a string, embeddings will be computed.

    If by_vector is provided, items are compared to by_vector instead of each other.
    """
    if not list_:
        return []

    embeddings_client = EmbeddingsClientSmall()
    reduced_list = []
    item_embeddings = []

    if rerank and isinstance(rerank, dict):
        rerank = Rerank(**rerank)

    for item in list_:
        if access_key:
            accessed_value = _get_nested_value(item, access_key)
            if accessed_value is None:
                continue

            if _is_vector(accessed_value):
                vector_item = list(accessed_value)
            else:
                text = str(accessed_value)
                vector_item = embeddings_client.embed_text(text)
        else:
            text = str(item)
            vector_item = embeddings_client.embed_text(text)
        if by_vector is not None:
            similarity = cosine_similarity(vector_item, by_vector)

            rerank_similarity = None
            if rerank and rerank.local:
                txt_value = _get_nested_value(item, rerank.local)
                if txt_value:
                    vector_txt_value = embeddings_client.embed_text(txt_value)
                    v_comp_txt_value = embeddings_client.embed_text(rerank.with_)
                    rerank_similarity = cosine_similarity(
                        vector_txt_value, v_comp_txt_value
                    )

            if similarity >= similarity_threshold or (
                rerank_similarity and rerank_similarity >= similarity_threshold
            ):
                reduced_list.append(item)
        else:
            is_similar = False

            for i, existing_vector in enumerate(item_embeddings):
                similarity = cosine_similarity(vector_item, existing_vector)

                rerank_similarity = None
                if rerank and rerank.local:
                    txt_value = _get_nested_value(item, rerank.local)
                    if txt_value:
                        vector_txt_value = embeddings_client.embed_text(txt_value)
                        v_comp_txt_value = embeddings_client.embed_text(rerank.with_)
                        rerank_similarity = cosine_similarity(
                            vector_txt_value, v_comp_txt_value
                        )

                if similarity >= similarity_threshold or (
                    rerank_similarity and rerank_similarity >= similarity_threshold
                ):
                    is_similar = True
                    break

            if not is_similar:
                reduced_list.append(item)
                item_embeddings.append(vector_item)

    return reduced_list
