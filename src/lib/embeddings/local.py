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

from sentence_transformers import SentenceTransformer

from src.adapters.interfaces.embeddings import EmbeddingsClient
from src.config import config

E5_PREFIX = "passage: "

_model = SentenceTransformer(config.embeddings.local_model)


class EmbeddingError(Exception):
    pass


class LocalEmbeddingsClient(EmbeddingsClient):
    def __init__(self):
        self.model = _model

    def _prefix(self, text: str) -> str:
        if text.strip().lower().startswith(E5_PREFIX.strip().lower()):
            return text
        return E5_PREFIX + text

    def embed_text(self, text: str) -> list[float]:
        try:
            prefixed = self._prefix(text)
            embedding = self.model.encode(prefixed, convert_to_numpy=True)
            return embedding.tolist()
        except Exception as e:
            raise EmbeddingError(f"Embedding failed: {e}") from e

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        try:
            prefixed = [self._prefix(t) for t in texts]
            embeddings = self.model.encode(prefixed, convert_to_numpy=True)
            return embeddings.tolist()
        except Exception as e:
            raise EmbeddingError(f"Embedding failed: {e}") from e


_embeddings_local_client = LocalEmbeddingsClient()
