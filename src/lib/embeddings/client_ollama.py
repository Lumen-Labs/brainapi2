from openai import OpenAI

from src.adapters.interfaces.embeddings import EmbeddingsClient
from src.config import config


class OllamaEmbeddingsClient(EmbeddingsClient):
    def __init__(self):
        self.client = OpenAI(
            base_url=f"http://{config.ollama.host}:{config.ollama.port}/v1/",
            api_key="ollama",
        )
        self.model = config.embeddings.local_model

    def embed_text(self, text: str) -> list[float]:
        response = self.client.embeddings.create(model=self.model, input=text)
        return response.data[0].embedding

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        response = self.client.embeddings.create(model=self.model, input=texts)
        sorted_data = sorted(response.data, key=lambda item: item.index)
        return [item.embedding for item in sorted_data]


_embeddings_ollama_client = OllamaEmbeddingsClient()
