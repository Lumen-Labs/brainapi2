"""
File: /client_small.py
Created Date: Wednesday December 31st 2025
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Wednesday December 31st 2025 9:55:46 am
Modified By: Christian Nonis <alch.infoemail@gmail.com>
-----
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.adapters.interfaces.embeddings import EmbeddingsClient
from src.config import config

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(config.embeddings.small_model)
    return _model


class EmbeddingsClientSmall(EmbeddingsClient):
    def embed_text(self, text: str) -> list[float]:
        """
        Return the embedding vector for the given text using the configured model.

        Parameters:
            text (str): Input text to encode.

        Returns:
            list[float]: Dense embedding vector representing the input text.
        """
        embedding = _get_model().encode(text, convert_to_numpy=True)
        return embedding.tolist()

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        Return the embedding vectors for the given texts using the configured model.

        Parameters:
            texts (list[str]): Input texts to encode.

        Returns:
            list[list[float]]: Dense embedding vectors representing the input texts.
        """
        embeddings = _get_model().encode(texts, convert_to_numpy=True)
        return embeddings.tolist()


_embeddings_small_client = EmbeddingsClientSmall()
