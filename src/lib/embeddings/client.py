"""
File: /client.py
Created Date: Friday October 24th 2025
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Friday October 24th 2025 7:11:17 pm
Modified By: the developer formerly known as Christian Nonis at <alch.infoemail@gmail.com>
-----
"""

import requests
import base64
import struct
from src.config import config

from src.adapters.interfaces.embeddings import EmbeddingsClient


class EmbeddingsClient(EmbeddingsClient):
    """
    Embeddings client.
    """

    def __init__(self):
        self.embeddings = None

    def embed_text(self, text: str) -> list[float]:
        """
        Embed a text and return a list of floats.
        """
        try:
            response = requests.post(
                config.azure.embedding_full_endpoint,
                json={"input": [text], "encoding_format": "base64"},
                headers={
                    "api-key": config.azure.embedding_key,
                    "Content-Type": "application/json",
                },
                timeout=60.0,
            )
            _b64_result = response.json()["data"][0]["embedding"]
            decoded_bytes = base64.b64decode(_b64_result)
            result = list(struct.unpack(f"<{len(decoded_bytes)//4}f", decoded_bytes))
            return result
        except Exception as e:
            print(f"Embedding encoding failed: {e}")
            raise


_embeddings_client = EmbeddingsClient()
