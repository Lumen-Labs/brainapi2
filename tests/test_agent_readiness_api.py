import os
import re
import unittest
from unittest.mock import AsyncMock, Mock, patch

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

from fastapi import FastAPI, HTTPException, Query
from fastapi.testclient import TestClient

from src.services.api.constants.requests import SearchResponse
from src.services.api.errors import install_error_handlers
from src.services.api.middlewares.auth import BrainPATMiddleware
from src.services.api.middlewares.brains import BrainMiddleware
from src.services.api.routes.public import public_router


def _public_app() -> FastAPI:
    app = FastAPI()
    install_error_handlers(app)
    app.include_router(public_router)
    return app


class StructuredErrorTests(unittest.TestCase):
    def test_http_error_preserves_detail_and_request_id(self):
        app = FastAPI()
        install_error_handlers(app)

        @app.get("/bad")
        async def bad():
            raise HTTPException(status_code=404, detail="missing")

        response = TestClient(app).get(
            "/bad",
            headers={"X-Request-ID": "request-123"},
        )
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["detail"], "missing")
        self.assertEqual(response.json()["error"]["code"], "RESOURCE_NOT_FOUND")
        self.assertTrue(response.json()["error"]["resolution"])
        self.assertEqual(response.json()["error"]["request_id"], "request-123")
        self.assertEqual(response.headers["X-Request-ID"], "request-123")

    def test_validation_error_keeps_structured_detail(self):
        app = FastAPI()
        install_error_handlers(app)

        @app.get("/validate")
        async def validate(value: int = Query(..., ge=1)):
            return {"value": value}

        response = TestClient(app).get("/validate?value=0")
        self.assertEqual(response.status_code, 422)
        self.assertIsInstance(response.json()["detail"], list)
        self.assertEqual(response.json()["error"]["code"], "VALIDATION_ERROR")

    def test_unexpected_error_is_sanitized(self):
        app = FastAPI()
        install_error_handlers(app)

        @app.get("/explode")
        async def explode():
            raise RuntimeError("secret backend detail")

        response = TestClient(app, raise_server_exceptions=False).get("/explode")
        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json()["detail"], "Internal server error")
        self.assertNotIn("secret backend detail", response.text)

    def test_unknown_route_and_brain_failure_use_stable_codes(self):
        app = FastAPI()
        install_error_handlers(app)

        @app.get("/brain")
        async def brain_failure():
            raise HTTPException(status_code=406, detail="unknown brain")

        missing = TestClient(app).get("/missing")
        brain = TestClient(app).get("/brain")
        self.assertEqual(missing.status_code, 404)
        self.assertEqual(missing.json()["error"]["code"], "RESOURCE_NOT_FOUND")
        self.assertEqual(brain.status_code, 406)
        self.assertEqual(brain.json()["error"]["code"], "BRAIN_NOT_FOUND")

    def test_auth_middleware_returns_structured_json(self):
        app = FastAPI()
        install_error_handlers(app)
        app.add_middleware(BrainPATMiddleware)

        @app.get("/system/test")
        async def protected():
            return {"ok": True}

        with patch.dict(os.environ, {"BRAINPAT_TOKEN": "system-token"}):
            response = TestClient(app).get("/system/test")
        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.json()["error"]["code"], "AUTH_INVALID")

    def test_brain_middleware_validates_before_backing_store_access(self):
        app = FastAPI()
        install_error_handlers(app)
        app.add_middleware(BrainMiddleware)

        @app.get("/protected")
        async def protected():
            return {"ok": True}

        response = TestClient(app).get(
            "/protected",
            headers={"BrainPAT": "invalid", "X-Brain-ID": "not-valid!"},
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"]["code"], "BRAIN_ID_INVALID")

    def test_brain_store_failure_is_structured_service_unavailable(self):
        app = FastAPI()
        install_error_handlers(app)
        app.add_middleware(BrainMiddleware)

        @app.get("/protected")
        async def protected():
            return {"ok": True}

        with patch(
            "src.services.api.middlewares.brains.cache_adapter.get",
            side_effect=ConnectionError("redis secret"),
        ):
            response = TestClient(app).get(
                "/protected",
                headers={"BrainPAT": "invalid", "X-Brain-ID": "validbrain"},
            )
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["error"]["code"], "SERVICE_UNAVAILABLE")
        self.assertNotIn("redis secret", response.text)


class PublicApiTests(unittest.TestCase):
    def test_health_is_typed_and_public(self):
        response = TestClient(_public_app()).get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")
        self.assertEqual(response.json()["service"], "brainapi")
        self.assertTrue(response.json()["version"])

    def test_full_app_exempts_machine_routes_and_authenticates_protected_routes(self):
        from src.services.api.app import app

        client = TestClient(app)
        health = client.get("/health", headers={"Origin": "https://agent.example"})
        openapi = client.get("/openapi.json")
        protected = client.get("/tasks/", headers={"X-Request-ID": "auth-check"})

        self.assertEqual(health.status_code, 200)
        self.assertNotIn("Access-Control-Allow-Origin", health.headers)
        self.assertEqual(openapi.status_code, 200)
        self.assertEqual(openapi.json()["openapi"], "3.1.0")
        self.assertEqual(protected.status_code, 401)
        self.assertEqual(protected.json()["error"]["code"], "AUTH_INVALID")
        self.assertEqual(protected.headers["X-Request-ID"], "auth-check")

    def test_demo_is_disabled_by_default(self):
        with patch.dict(os.environ, {"PUBLIC_DEMO_ENABLED": "false"}):
            response = TestClient(_public_app()).get("/demo/search?query=memory")
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["error"]["code"], "RESOURCE_NOT_FOUND")

    def test_demo_forces_brain_passages_and_k_limit(self):
        captured = []

        async def fake_search(request):
            captured.append(request)
            return SearchResponse(hits=[])

        env = {
            "PUBLIC_DEMO_ENABLED": "true",
            "PUBLIC_DEMO_BRAIN_ID": "agentdemo",
            "PUBLIC_DEMO_MAX_K": "5",
        }
        with (
            patch.dict(os.environ, env),
            patch("src.services.api.routes.public.config.search_enabled", True),
            patch("src.services.api.routes.public.config.search_use_bm25", True),
            patch("src.services.api.routes.public.config.search_use_dense", False),
            patch(
                "src.services.api.routes.public.search_controller",
                new=AsyncMock(side_effect=fake_search),
            ),
        ):
            response = TestClient(_public_app()).get(
                "/demo/search?query=memory&k=5&brain_id=customerbrain"
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(captured), 1)
        self.assertEqual(captured[0].brain_id, "agentdemo")
        self.assertEqual(captured[0].channels, ["passages"])
        self.assertEqual(captured[0].k, 5)

    def test_demo_rejects_configured_limit_and_long_query(self):
        env = {
            "PUBLIC_DEMO_ENABLED": "true",
            "PUBLIC_DEMO_BRAIN_ID": "agentdemo",
            "PUBLIC_DEMO_MAX_K": "3",
        }
        with patch.dict(os.environ, env):
            too_many = TestClient(_public_app()).get("/demo/search?query=x&k=4")
            too_long = TestClient(_public_app()).get(
                f"/demo/search?query={'x' * 501}"
            )
        self.assertEqual(too_many.status_code, 422)
        self.assertEqual(too_long.status_code, 422)

    def test_demo_reports_unavailable_search_dependency(self):
        env = {
            "PUBLIC_DEMO_ENABLED": "true",
            "PUBLIC_DEMO_BRAIN_ID": "agentdemo",
            "PUBLIC_DEMO_MAX_K": "10",
        }
        with (
            patch.dict(os.environ, env),
            patch("src.services.api.routes.public.config.search_enabled", False),
        ):
            response = TestClient(_public_app()).get("/demo/search?query=memory")

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["error"]["code"], "SERVICE_UNAVAILABLE")
        self.assertTrue(response.json()["error"]["resolution"])

    def test_demo_controller_failures_are_structured_and_writes_are_absent(self):
        env = {
            "PUBLIC_DEMO_ENABLED": "true",
            "PUBLIC_DEMO_BRAIN_ID": "agentdemo",
            "PUBLIC_DEMO_MAX_K": "10",
        }
        with (
            patch.dict(os.environ, env),
            patch("src.services.api.routes.public.config.search_enabled", True),
            patch("src.services.api.routes.public.config.search_use_bm25", True),
            patch("src.services.api.routes.public.config.search_use_dense", False),
            patch(
                "src.services.api.routes.public.search_controller",
                new=AsyncMock(side_effect=RuntimeError("backend secret")),
            ),
        ):
            failed = TestClient(
                _public_app(), raise_server_exceptions=False
            ).get("/demo/search?query=memory")
        write = TestClient(_public_app()).post("/demo/search", json={"query": "x"})

        self.assertEqual(failed.status_code, 500)
        self.assertEqual(failed.json()["error"]["code"], "INTERNAL_ERROR")
        self.assertNotIn("backend secret", failed.text)
        self.assertEqual(write.status_code, 405)
        self.assertEqual(write.json()["error"]["code"], "METHOD_NOT_ALLOWED")
        self.assertEqual(write.headers["Allow"], "GET")


class OpenApiContractTests(unittest.TestCase):
    def test_operations_are_function_calling_ready(self):
        from src.services.api.app import app

        app.openapi_schema = None
        schema = app.openapi()
        operations = []
        for path, path_item in schema["paths"].items():
            for method, operation in path_item.items():
                if method not in {"get", "post", "put", "patch", "delete"}:
                    continue
                operations.append((path, method, operation))

        operation_ids = [operation["operationId"] for _, _, operation in operations]
        self.assertEqual(len(operation_ids), len(set(operation_ids)))
        for path, method, operation in operations:
            operation_id = operation["operationId"]
            self.assertRegex(operation_id, re.compile(r"^[a-z][a-z0-9_]{0,63}$"))
            self.assertTrue(operation.get("description"), (method, path))
            for parameter in operation.get("parameters", []):
                self.assertTrue(
                    "$ref" in parameter or "schema" in parameter,
                    (method, path, parameter),
                )
            success_schemas = [
                media.get("schema")
                for status, response in operation["responses"].items()
                if str(status).startswith("2")
                for media in response.get("content", {}).values()
            ]
            self.assertTrue(success_schemas, (method, path))
            self.assertTrue(all(item for item in success_schemas), (method, path))
            expected_security = [] if path in {"/health", "/demo/search"} else [
                {"BrainPAT": []},
                {"BearerAuth": []},
            ]
            self.assertEqual(operation.get("security"), expected_security)
            if path not in {"/health", "/demo/search"}:
                self.assertEqual(
                    operation["responses"]["422"]["content"]["application/json"][
                        "schema"
                    ]["$ref"],
                    "#/components/schemas/ErrorResponse",
                )

        self.assertIn("/system/brains/{brain_id}/clear", schema["paths"])
        self.assertIn("post", schema["paths"]["/system/brains/{brain_id}/clear"])
        self.assertIn("/system/brains/{brain_id}", schema["paths"])
        self.assertIn("delete", schema["paths"]["/system/brains/{brain_id}"])
        self.assertIn("202", schema["paths"]["/ingest/file"]["post"]["responses"])
        self.assertIn("202", schema["paths"]["/ingest/url"]["post"]["responses"])
        self.assertIn(
            "503",
            schema["paths"]["/ingest/url"]["post"]["responses"],
        )

        generic_refs = [
            response["content"]["application/json"]["schema"].get("$ref", "")
            for _, _, operation in operations
            for status, response in operation["responses"].items()
            if str(status).startswith("2")
            and "application/json" in response.get("content", {})
        ]
        self.assertFalse(any("Generic" in ref for ref in generic_refs))


class PublicDemoSeedTests(unittest.TestCase):
    def test_seed_command_skips_an_up_to_date_brain(self):
        from scripts import seed_public_demo

        docs = Mock()
        docs.text = "public docs"
        docs.raise_for_status.return_value = None
        existing = Mock(status_code=200)
        existing.json.return_value = {"total": 1}
        post = Mock()

        with (
            patch.dict(os.environ, {"BRAINPAT_TOKEN": "secret", "PUBLIC_DEMO_BRAIN_ID": "agentdemo"}),
            patch("sys.argv", ["seed_public_demo.py"]),
            patch.object(seed_public_demo.requests, "get", side_effect=[docs, existing]),
            patch.object(seed_public_demo.requests, "post", post),
        ):
            result = seed_public_demo.main()

        self.assertEqual(result, 0)
        post.assert_not_called()


if __name__ == "__main__":
    unittest.main()
