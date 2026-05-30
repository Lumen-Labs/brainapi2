import json

import boto3

from src.adapters.interfaces.embeddings import EmbeddingsClient
from src.config import config


class BedrockEmbeddingsClient(EmbeddingsClient):
    def __init__(self):
        session = boto3.Session(
            aws_access_key_id=config.bedrock.access_key_id,
            aws_secret_access_key=config.bedrock.secret_access_key,
            aws_session_token=config.bedrock.session_token,
            region_name=config.bedrock.region,
        )
        self.client = session.client("bedrock-runtime")
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
