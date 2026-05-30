import os

from google import genai

from src.adapters.interfaces.embeddings import EmbeddingsClient
from src.config import config


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
        response = self.client.models.embed_content(model=self.model, contents=[text])
        return response.embeddings[0].values

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        response = self.client.models.embed_content(model=self.model, contents=texts)
        return [item.values for item in response.embeddings]


_embeddings_vertex_client = VertexEmbeddingsClient()
