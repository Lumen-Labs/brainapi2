"""
File: /client_small.py
Created Date: Wednesday December 31st 2025
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Wednesday December 31st 2025 9:55:46 am
Modified By: Christian Nonis <alch.infoemail@gmail.com>
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
        """
        Return the embedding vector for the given text using the configured model.
        
        Parameters:
            text (str): Input text to encode.
        
        Returns:
            list[float]: Dense embedding vector representing the input text.
        """
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()


_embeddings_small_client = EmbeddingsClientSmall()