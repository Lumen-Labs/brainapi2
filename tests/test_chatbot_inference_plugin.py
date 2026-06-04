import os
import sys
import unittest
from pathlib import Path

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
}
for key, value in ENV_DEFAULTS.items():
    os.environ.setdefault(key, value)

CHATBOT_PLUGIN_DIR = Path(__file__).resolve().parent.parent / "plugins" / "chatbot"
if str(CHATBOT_PLUGIN_DIR) not in sys.path:
    sys.path.insert(0, str(CHATBOT_PLUGIN_DIR))

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from main import register
from routes import inference as inference_routes
from adapters.inference import parse_model
from src.core.plugins.context import PluginContext


class ChatbotInferenceAdapterTests(unittest.TestCase):
    def test_parse_model_valid(self):
        provider, model = parse_model("azureopenai::gpt-4o-mini")
        self.assertEqual(provider, "azure")
        self.assertEqual(model, "gpt-4o-mini")

    def test_parse_model_invalid_format(self):
        with self.assertRaises(ValueError):
            parse_model("azure-gpt-4o-mini")

    def test_parse_model_unsupported_provider(self):
        with self.assertRaises(ValueError):
            parse_model("unknown::model-a")


class ChatbotInferenceRouteTests(unittest.TestCase):
    def setUp(self):
        self.app = FastAPI()

        @self.app.middleware("http")
        async def set_brain_id(request: Request, call_next):
            request.state.brain_id = "brain-test"
            return await call_next(request)

        self.app.include_router(inference_routes.create_inference_router(PluginContext))
        self.client = TestClient(self.app)
        self.original_inference_response = inference_routes.inference_response

    def tearDown(self):
        inference_routes.inference_response = self.original_inference_response

    async def _fake_inference_response(
        self, model, prompt, stream=False, max_tokens=None, **kwargs
    ):
        self.assertFalse(stream)
        self.assertEqual(model, "openai::gpt-4o-mini")
        self.assertIn("hello", prompt)
        self.assertEqual(kwargs.get("brain_id"), "brain-test")
        self.assertEqual(max_tokens, 16)
        return "openai", "gpt-4o-mini", "ok-result"

    def test_inference_non_stream_success(self):
        inference_routes.inference_response = self._fake_inference_response
        response = self.client.post(
            "/chatbot/inference",
            json={
                "model": "openai::gpt-4o-mini",
                "input": "hello",
                "stream": False,
                "max_tokens": 16,
            },
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["data"]["provider"], "openai")
        self.assertEqual(payload["data"]["output"], "ok-result")
        self.assertFalse(payload["data"]["stream"])

    def test_inference_invalid_model_returns_422(self):
        async def fake_inference_response(model, prompt, stream=False, max_tokens=None, **kwargs):
            raise ValueError("model must be in 'api_provider::llm_name' format")

        inference_routes.inference_response = fake_inference_response
        response = self.client.post(
            "/chatbot/inference",
            json={"model": "bad-format", "input": "hello", "stream": False},
        )
        self.assertEqual(response.status_code, 422)
        self.assertIn("api_provider::llm_name", response.json()["detail"])

    def test_inference_stream_success(self):
        async def fake_event_stream():
            yield 'data: {"choices":[{"delta":{"content":"hello"}}]}\n\n'
            yield "data: [DONE]\n\n"

        async def fake_inference_response(model, prompt, stream=False, max_tokens=None, **kwargs):
            self.assertTrue(stream)
            return "openai", "gpt-4o-mini", fake_event_stream()

        inference_routes.inference_response = fake_inference_response
        response = self.client.post(
            "/chatbot/inference",
            json={"model": "openai::gpt-4o-mini", "input": "hello", "stream": True},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.headers.get("content-type").split(";")[0], "text/event-stream"
        )
        body = response.text
        self.assertIn("data: {", body)
        self.assertIn("data: [DONE]", body)


class ChatbotInferenceRegisterTests(unittest.TestCase):
    class DummyContext:
        def __init__(self):
            self._app = object()
            self.routers = []

        def include_router(self, router, **kwargs):
            self.routers.append((router, kwargs))

    def test_register_adds_chatbot_router(self):
        context = self.DummyContext()
        register(context)
        self.assertEqual(len(context.routers), 1)
        router, _ = context.routers[0]
        self.assertEqual(router.prefix, "/chatbot")


if __name__ == "__main__":
    unittest.main()
