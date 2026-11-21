import asyncio
import os
from unittest.mock import MagicMock, patch

ENV_DEFAULTS = {
    "BRAINPAT_TOKEN": "test-token",
    "MODELS_MODE": "local",
    "EMBEDDINGS_LOCAL_MODEL": "local-model",
    "EMBEDDINGS_SMALL_MODEL": "small-model",
    "EMBEDDING_NODES_DIMENSION": "3",
    "EMBEDDING_TRIPLETS_DIMENSION": "3",
    "EMBEDDING_OBSERVATIONS_DIMENSION": "3",
    "EMBEDDING_DATA_DIMENSION": "3",
    "EMBEDDING_RELATIONSHIPS_DIMENSION": "3",
    "REDIS_HOST": "localhost",
    "REDIS_PORT": "6379",
    "NEO4J_HOST": "localhost",
    "NEO4J_PORT": "7687",
    "NEO4J_USERNAME": "neo4j",
    "NEO4J_PASSWORD": "password",
    "MILVUS_HOST": "localhost",
    "MILVUS_PORT": "19530",
    "MONGO_CONNECTION_STRING": "mongodb://localhost:27017",
    "CELERY_WORKER_CONCURRENCY": "1",
    "OLLAMA_HOST": "localhost",
    "OLLAMA_PORT": "11434",
    "OLLAMA_LLM_SMALL_MODEL": "small",
    "OLLAMA_LLM_LARGE_MODEL": "large",
    "PIPELINE_MODE": "accurate",
}
for key, value in ENV_DEFAULTS.items():
    os.environ.setdefault(key, value)

import pytest
from fastapi import FastAPI, HTTPException, Request
from fastapi.testclient import TestClient

from src.adapters.embeddings import (
    EmbeddingError,
    EmbeddingsAdapter,
    RaiseEmbeddingFailureStrategy,
)
from src.constants.embeddings import Vector
from src.core.search.entity_context import EntityContext
from src.lib.embeddings.client import EmbeddingClientError
from src.services.api.controllers.entities import get_entity_context
from src.services.api.controllers.model import add_nodes
from src.services.api.controllers.retrieve import _embed_query_vector
from src.services.api.controllers.vectors import list_vectors
from src.services.api.errors import install_error_handlers
from src.services.api.routes.model import model_router
from src.services.api.routes.retrieve import retrieve_router


class ClientError(Exception):
    pass


def _app() -> FastAPI:
    app = FastAPI()
    install_error_handlers(app)

    @app.middleware("http")
    async def inject_brain(request: Request, call_next):
        request.state.brain_id = "testbrain"
        return await call_next(request)

    app.include_router(model_router)
    app.include_router(retrieve_router)
    return app


def test_add_entity_empty_name_returns_422():
    response = TestClient(_app()).post("/model/entity", json={"name": ""})
    assert response.status_code == 422
    body = response.json()
    assert body["error"]["code"] == "VALIDATION_ERROR"
    assert "Internal server error" not in response.text
    assert "Traceback" not in response.text


def test_add_entity_missing_name_returns_422():
    response = TestClient(_app()).post("/model/entity", json={"labels": ["PERSON"]})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"


def test_add_nodes_empty_name_is_http_422():
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(add_nodes([{"name": ""}], brain_id="testbrain"))
    assert exc_info.value.status_code == 422
    assert "empty" in str(exc_info.value.detail).lower()


def test_unknown_vector_store_is_http_404():
    with patch(
        "src.services.api.controllers.vectors.vector_store_adapter"
    ) as adapter:
        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(list_vectors("not-a-store", "testbrain"))
    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == "Vector store not found"
    adapter.list_vectors.assert_not_called()


def test_unknown_vector_store_route_returns_404():
    with patch(
        "src.services.api.controllers.vectors.vector_store_adapter.list_vectors"
    ) as list_fn:
        response = TestClient(_app()).get("/retrieve/vectors/not-a-store")
    assert response.status_code == 404
    body = response.json()
    assert body["detail"] == "Vector store not found"
    assert body["error"]["code"] == "RESOURCE_NOT_FOUND"
    assert "Traceback" not in response.text
    list_fn.assert_not_called()


def test_adapter_maps_client_error_without_leaking():
    class FakeClient:
        def embed_text(self, text):
            raise ClientError("secret upstream detail")

        def embed_texts(self, texts):
            raise ClientError("secret upstream detail")

    adapter = EmbeddingsAdapter()
    adapter.add_client(FakeClient())
    with pytest.raises(EmbeddingClientError) as exc_info:
        adapter.embed_text("hello")
    assert str(exc_info.value) == "Embedding request failed"
    assert "secret upstream detail" not in str(exc_info.value)
    with pytest.raises(EmbeddingClientError) as batch_info:
        adapter.embed_texts(["a", "b"])
    assert str(batch_info.value) == "Embedding request failed"
    assert "secret upstream detail" not in str(batch_info.value)


def test_adapter_raise_strategy_maps_client_error():
    class FakeClient:
        def embed_text(self, text):
            raise ClientError("secret upstream detail")

    adapter = EmbeddingsAdapter()
    adapter.add_client(FakeClient())
    adapter.set_failure_strategy(RaiseEmbeddingFailureStrategy())
    with pytest.raises(EmbeddingClientError) as exc_info:
        adapter.embed_text("hello")
    assert "secret upstream detail" not in str(exc_info.value)
    assert str(exc_info.value) == "Embedding request failed"


def test_adapter_default_strategy_still_returns_empty_on_embedding_error():
    adapter = EmbeddingsAdapter()
    adapter._embed_text_with_retry = MagicMock(side_effect=EmbeddingError("failed"))
    adapter._embed_texts_with_retry = MagicMock(side_effect=EmbeddingError("failed"))
    vector = adapter.embed_text("hello")
    assert vector.embeddings == []
    many = adapter.embed_texts(["a", "b"])
    assert len(many) == 2
    assert all(item.embeddings == [] for item in many)


def test_embed_query_vector_maps_adapter_client_error_to_503():
    class FakeClient:
        def embed_text(self, text):
            raise ClientError("secret upstream detail")

    adapter = EmbeddingsAdapter()
    adapter.add_client(FakeClient())
    with patch(
        "src.services.api.controllers.retrieve.embeddings_adapter",
        adapter,
    ):
        with pytest.raises(HTTPException) as exc_info:
            _embed_query_vector("hello")
    assert exc_info.value.status_code == 503
    assert exc_info.value.detail == "Embedding service unavailable"
    assert "secret upstream detail" not in str(exc_info.value.detail)


def test_embed_query_vector_returns_503_on_client_failure():
    failing = MagicMock()
    failing.embed_text.side_effect = ClientError("secret upstream detail")
    with patch(
        "src.services.api.controllers.retrieve.embeddings_adapter",
        failing,
    ):
        with pytest.raises(HTTPException) as exc_info:
            _embed_query_vector("hello")
    assert exc_info.value.status_code == 503
    assert exc_info.value.detail == "Embedding service unavailable"


def test_embed_query_vector_returns_503_on_empty_vector():
    empty = MagicMock()
    empty.embed_text.return_value = Vector(id="q", embeddings=[], metadata={})
    with patch(
        "src.services.api.controllers.retrieve.embeddings_adapter",
        empty,
    ):
        with pytest.raises(HTTPException) as exc_info:
            _embed_query_vector("hello")
    assert exc_info.value.status_code == 503
    assert exc_info.value.detail == "Embedding service unavailable"


def test_embed_query_vector_http_envelope_is_503():
    app = FastAPI()
    install_error_handlers(app)

    @app.get("/embed")
    async def embed():
        return _embed_query_vector("hello")

    failing = MagicMock()
    failing.embed_text.side_effect = ClientError("secret upstream detail")
    with patch(
        "src.services.api.controllers.retrieve.embeddings_adapter",
        failing,
    ):
        response = TestClient(app, raise_server_exceptions=False).get("/embed")
    assert response.status_code == 503
    body = response.json()
    assert body["detail"] == "Embedding service unavailable"
    assert body["error"]["code"] == "SERVICE_UNAVAILABLE"
    assert "secret upstream detail" not in response.text
    assert "Traceback" not in response.text
    assert "ClientError" not in response.text


def test_embedding_client_error_handler_returns_503():
    app = FastAPI()
    install_error_handlers(app)

    @app.get("/embed-fail")
    async def embed_fail():
        raise EmbeddingClientError("secret upstream detail")

    response = TestClient(app, raise_server_exceptions=False).get("/embed-fail")
    assert response.status_code == 503
    body = response.json()
    assert body["detail"] == "Embedding service unavailable"
    assert "secret upstream detail" not in response.text
    assert "Traceback" not in response.text


def test_entity_context_missing_target_returns_four_tuple():
    with patch(
        "src.core.search.entity_context.resolve_entity_node",
        return_value=None,
    ):
        result = EntityContext("missing", "testbrain").get_context()
    assert result == (None, [], [], [])


def test_get_entity_context_missing_entity_is_http_404():
    with patch(
        "src.services.api.controllers.entities.EntityContext.get_context",
        return_value=(None, [], [], []),
    ):
        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(get_entity_context("missing", brain_id="testbrain"))
    assert exc_info.value.status_code == 404
    assert "No entity found" in exc_info.value.detail


def test_entity_context_route_returns_404():
    with patch(
        "src.services.api.controllers.entities.EntityContext.get_context",
        return_value=(None, [], [], []),
    ):
        response = TestClient(_app()).get(
            "/retrieve/entity/context",
            params={"target": "missing"},
        )
    assert response.status_code == 404
    body = response.json()
    assert body["detail"] == "No entity found matching the target."
    assert body["error"]["code"] == "RESOURCE_NOT_FOUND"
    assert "Traceback" not in response.text
    assert "ValueError" not in response.text
