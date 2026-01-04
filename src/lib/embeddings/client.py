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
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    RetryError,
)
from requests.exceptions import ConnectionError, Timeout, RequestException
from src.config import config

from src.adapters.interfaces.embeddings import EmbeddingsClient


class EmbeddingError(Exception):
    """Custom exception for embedding failures after retries."""

    pass


class EmbeddingsClient(EmbeddingsClient):
    """
    Embeddings client.
    """

    def __init__(self):
        self.embeddings = None

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type((ConnectionError, Timeout)),
        reraise=False,
    )
    def _embed_text_with_retry(self, text: str) -> list[float]:
        """
        Internal method to embed text with retry logic for network errors.
        """
        response = requests.post(
            config.azure.embedding_full_endpoint,
            json={"input": [text], "encoding_format": "base64"},
            headers={
                "api-key": config.azure.embedding_key,
                "Content-Type": "application/json",
            },
            timeout=60.0,
        )
        response.raise_for_status()
        _b64_result = response.json()["data"][0]["embedding"]
        decoded_bytes = base64.b64decode(_b64_result)
        result = list(struct.unpack(f"<{len(decoded_bytes)//4}f", decoded_bytes))
        return result

    def embed_text(self, text: str) -> list[float]:
        """
        Embed a text and return a list of floats.
        Handles connection errors with retry logic.
        """
        try:
            return self._embed_text_with_retry(text)
        except RetryError as e:
            last_attempt = e.last_attempt
            error_msg = (
                f"Embedding failed after {last_attempt.attempt_number} attempts. "
                f"Last error: {last_attempt.exception()}"
            )
            print(f"Embedding encoding failed: {error_msg}")
            raise EmbeddingError(error_msg) from last_attempt.exception()
        except (ConnectionError, Timeout) as e:
            error_msg = f"Network error during embedding: {e}"
            print(f"Embedding encoding failed: {error_msg}")
            raise EmbeddingError(error_msg) from e
        except RequestException as e:
            error_msg = f"Request error during embedding: {e}"
            print(f"Embedding encoding failed: {error_msg}")
            raise EmbeddingError(error_msg) from e
        except Exception as e:
            error_msg = f"Unexpected error during embedding: {e}"
            print(f"Embedding encoding failed: {error_msg}")
            raise EmbeddingError(error_msg) from e


_embeddings_client = EmbeddingsClient()
