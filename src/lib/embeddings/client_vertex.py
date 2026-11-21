import os

from google import genai

from src.adapters.interfaces.embeddings import EmbeddingsClient
from src.config import config
from src.lib.embeddings.client import EmbeddingClientError


class VertexEmbeddingsClient(EmbeddingsClient):
    def __init__(self):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = config.gcp.credentials_path
        self.client = genai.Client(
            vertexai=True,
            project=config.gcp.project_id,
            location="global",
            http_options={"api_version": "v1"},
        )
        self.model = config.gcp.embedding_model

    def embed_text(self, text: str) -> list[float]:
        try:
            response = self.client.models.embed_content(
                model=self.model, contents=[text]
            )
            return response.embeddings[0].values
        except EmbeddingClientError:
            raise
        except Exception as exc:
            raise EmbeddingClientError("Embedding request failed") from exc

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        try:
            response = self.client.models.embed_content(
                model=self.model, contents=texts
            )
            return [item.values for item in response.embeddings]
        except EmbeddingClientError:
            raise
        except Exception as exc:
            raise EmbeddingClientError("Embedding request failed") from exc


_embeddings_vertex_client = VertexEmbeddingsClient()
