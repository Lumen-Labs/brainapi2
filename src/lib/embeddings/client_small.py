"""
File: /client_small.py
Created Date: Wednesday December 31st 2025
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Wednesday December 31st 2025 9:55:46 am
Modified By: the developer formerly known as Christian Nonis at <alch.infoemail@gmail.com>
-----
"""

from sentence_transformers import SentenceTransformer

from src.adapters.interfaces.embeddings import EmbeddingsClient
from src.config import config

_model = SentenceTransformer(config.embeddings.small_model)


class EmbeddingsClientSmall(EmbeddingsClient):
    def __init__(self):
        self.model = _model

    def embed_text(self, text: str) -> list[float]:
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()
