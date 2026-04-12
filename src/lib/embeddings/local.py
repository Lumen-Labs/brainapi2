"""
File: /local.py
Project: embeddings
Created Date: Sunday February 22nd 2026
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Sunday February 22nd 2026 5:24:22 pm
Modified By: Christian Nonis <alch.infoemail@gmail.com>
-----
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.adapters.interfaces.embeddings import EmbeddingsClient
from src.config import config

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

E5_PREFIX = "passage: "

_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(config.embeddings.local_model)
    return _model


class EmbeddingError(Exception):
    pass


class LocalEmbeddingsClient(EmbeddingsClient):
    def _prefix(self, text: str) -> str:
        if text.strip().lower().startswith(E5_PREFIX.strip().lower()):
            return text
        return E5_PREFIX + text

    def embed_text(self, text: str) -> list[float]:
        try:
            prefixed = self._prefix(text)
            embedding = _get_model().encode(prefixed, convert_to_numpy=True)
            return embedding.tolist()
        except Exception as e:
            raise EmbeddingError(f"Embedding failed: {e}") from e

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        try:
            prefixed = [self._prefix(t) for t in texts]
            embeddings = _get_model().encode(prefixed, convert_to_numpy=True)
            return embeddings.tolist()
        except Exception as e:
            raise EmbeddingError(f"Embedding failed: {e}") from e


_embeddings_local_client = LocalEmbeddingsClient()
