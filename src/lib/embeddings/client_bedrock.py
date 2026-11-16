import json

import boto3

from src.adapters.interfaces.embeddings import EmbeddingsClient
from src.config import config


class BedrockEmbeddingsClient(EmbeddingsClient):
    def __init__(self):
        session = boto3.Session(**config.bedrock.session_kwargs())
        self.client = session.client(
            "bedrock-runtime", **config.bedrock.runtime_client_kwargs()
        )
        self.model = config.bedrock.embedding_model

    def _embed_one(self, text: str) -> list[float]:
        body = json.dumps({"inputText": text})
        response = self.client.invoke_model(
            modelId=self.model,
            body=body,
            contentType="application/json",
            accept="application/json",
        )
        payload = json.loads(response["body"].read())
        return payload["embedding"]

    def embed_text(self, text: str) -> list[float]:
        return self._embed_one(text)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]


_embeddings_bedrock_client = BedrockEmbeddingsClient()
