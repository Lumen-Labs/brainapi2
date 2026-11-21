import inspect
import os
import unittest
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

from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel, ValidationError, field_validator

from src.constants.kg import Node
from src.core.search.entity_info import EventSynergyRetriever, MatchPath
from src.core.search.entity_resolve import node_matches_target, resolve_entity_node
from src.lib.cypher_safety import InvalidCypherIdentifier, require_label, require_labels
from src.services.api.constants.requests import (
    AddEntityRequest,
    AddRelationshipRequest,
    GetContextRequestBody,
)
from src.services.api.errors import install_error_handlers
from src.services.api.openapi import install_openapi_contract, openapi_servers


class OpenApiServerAndBrainIdTests(unittest.TestCase):
    def test_default_server_is_localhost(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("OPENAPI_SERVER_URL", None)
            os.environ.pop("OPENAPI_SERVER_URLS", None)
            servers = openapi_servers()
        self.assertEqual(servers[0]["url"], "http://localhost:8000")

    def test_env_overrides_server_url(self):
        with patch.dict(
            os.environ,
            {"OPENAPI_SERVER_URL": "https://api.example.test"},
            clear=False,
        ):
            os.environ.pop("OPENAPI_SERVER_URLS", None)
            servers = openapi_servers()
        self.assertEqual(servers[0]["url"], "https://api.example.test")

    def test_server_list_leads_when_single_url_unset(self):
        with patch.dict(
            os.environ,
            {"OPENAPI_SERVER_URLS": "https://eu.example.test,https://us.example.test"},
            clear=False,
        ):
            os.environ.pop("OPENAPI_SERVER_URL", None)
            servers = openapi_servers()
        self.assertEqual(
            [item["url"] for item in servers],
            ["https://eu.example.test", "https://us.example.test"],
        )

    def test_single_url_leads_and_dedupes_server_list(self):
        with patch.dict(
            os.environ,
            {
                "OPENAPI_SERVER_URL": "https://us.example.test",
                "OPENAPI_SERVER_URLS": "https://eu.example.test,https://us.example.test",
            },
            clear=False,
        ):
            servers = openapi_servers()
        self.assertEqual(
            [item["url"] for item in servers],
            ["https://us.example.test", "https://eu.example.test"],
        )

    def test_untyped_core_route_does_not_raise(self):
        app = FastAPI(title="BrainAPI", version="0.0.0")
        install_error_handlers(app)

        @app.get("/retrieve/untyped", tags=["retrieve"])
        def untyped():
            return {}

        install_openapi_contract(app)
        schema = app.openapi()
        self.assertIn("/retrieve/untyped", schema["paths"])
        media = schema["paths"]["/retrieve/untyped"]["get"]["responses"]["200"][
            "content"
        ]["application/json"]
        self.assertEqual(
            media.get("schema"),
            {"$ref": "#/components/schemas/GenericValueResponse"},
        )
        value_schema = schema["components"]["schemas"]["GenericValueResponse"]
        self.assertNotIn("type", value_schema)
        self.assertTrue(value_schema.get("description"))

    def test_untyped_list_path_keeps_array_schema(self):
        app = FastAPI(title="BrainAPI", version="0.0.0")
        install_error_handlers(app)

        @app.get("/retrieve/hops", tags=["retrieve"])
        def hops():
            return []

        install_openapi_contract(app)
        schema = app.openapi()
        media = schema["paths"]["/retrieve/hops"]["get"]["responses"]["200"]["content"][
            "application/json"
        ]
        self.assertEqual(
            media.get("schema"),
            {"$ref": "#/components/schemas/GenericListResponse"},
        )
        self.assertEqual(
            schema["components"]["schemas"]["GenericListResponse"]["type"],
            "array",
        )


def _validation_app() -> FastAPI:
    app = FastAPI()
    install_error_handlers(app)

    @app.post("/model/entity")
    async def add_entity(body: AddEntityRequest):
        return {"name": body.name}

    @app.post("/retrieve/context")
    async def get_context(body: GetContextRequestBody):
        return {"text": body.text}

    @app.post("/model/relationship")
    async def add_relationship(body: AddRelationshipRequest):
        return {"predicate_name": body.predicate_name}

    class BlankNameBody(BaseModel):
        name: str

        @field_validator("name")
        @classmethod
        def _name_not_blank(cls, value: str) -> str:
            text = value.strip()
            if not text:
                raise ValueError("Node name is required and cannot be empty")
            return text

    @app.post("/value-error")
    async def value_error_route(body: BlankNameBody):
        return {"name": body.name}

    return app


def _assert_422_json(test, response):
    test.assertEqual(response.status_code, 422)
    test.assertIn("application/json", response.headers.get("content-type", ""))
    body = response.json()
    test.assertIsInstance(body.get("detail"), list)
    test.assertEqual(body["error"]["code"], "VALIDATION_ERROR")
    test.assertNotIn("Internal server error", response.text)
    test.assertNotIn("Traceback", response.text)


class RequestValidationTests(unittest.TestCase):
    def test_empty_entity_name_is_rejected(self):
        with self.assertRaises(ValidationError):
            AddEntityRequest(name="", labels=["PERSON"])
        with self.assertRaises(ValidationError):
            AddEntityRequest(name="   ", labels=["PERSON"])

    def test_empty_context_text_is_rejected(self):
        with self.assertRaises(ValidationError):
            GetContextRequestBody(text="")
        with self.assertRaises(ValidationError):
            GetContextRequestBody(text="   ")

    def test_empty_predicate_name_is_rejected(self):
        with self.assertRaises(ValidationError):
            AddRelationshipRequest(
                subject_uuid="s",
                predicate_name="",
                predicate_description="d",
                object_uuid="o",
            )
        with self.assertRaises(ValidationError):
            AddRelationshipRequest(
                subject_uuid="s",
                predicate_name="   ",
                predicate_description="d",
                object_uuid="o",
            )

    def test_empty_and_whitespace_name_http_is_422_json(self):
        client = TestClient(_validation_app(), raise_server_exceptions=False)
        for payload in ({"name": ""}, {"name": "   "}):
            with self.subTest(payload=payload):
                _assert_422_json(self, client.post("/model/entity", json=payload))

    def test_empty_and_whitespace_text_http_is_422_json(self):
        client = TestClient(_validation_app(), raise_server_exceptions=False)
        for payload in ({"text": ""}, {"text": "   "}):
            with self.subTest(payload=payload):
                _assert_422_json(self, client.post("/retrieve/context", json=payload))

    def test_whitespace_predicate_name_http_is_422_json(self):
        client = TestClient(_validation_app(), raise_server_exceptions=False)
        response = client.post(
            "/model/relationship",
            json={
                "subject_uuid": "s",
                "predicate_name": "   ",
                "predicate_description": "d",
                "object_uuid": "o",
            },
        )
        _assert_422_json(self, response)

    def test_value_error_validator_serializes_to_422_json(self):
        client = TestClient(_validation_app(), raise_server_exceptions=False)
        response = client.post("/value-error", json={"name": "   "})
        _assert_422_json(self, response)
        self.assertEqual(response.status_code, 422)

    def test_require_label_rejects_injection(self):
        with self.assertRaises(InvalidCypherIdentifier):
            require_label("PERSON;DROP")
        with self.assertRaises(InvalidCypherIdentifier):
            require_label("PERSON`DETACH DELETE")

    def test_require_label_accepts_normal_labels(self):
        self.assertEqual(require_label("Software Engineer"), "Software Engineer")
        self.assertEqual(require_label("foo-bar"), "foo-bar")
        self.assertEqual(require_label("Café"), "Café")
        self.assertEqual(require_label("person"), "person")
        self.assertEqual(require_labels([]), [])
        self.assertEqual(
            require_labels(["Software Engineer", "foo-bar"]),
            ["Software Engineer", "foo-bar"],
        )


class EntityResolveTests(unittest.TestCase):
    def test_node_matches_uuid_and_name(self):
        node = Node(uuid="abc-1", name="Alice", labels=["PERSON"])
        self.assertTrue(node_matches_target(node, "abc-1"))
        self.assertTrue(node_matches_target(node, "alice"))
        self.assertFalse(node_matches_target(node, "Alicia"))
        self.assertFalse(node_matches_target(None, "Alice"))

    def test_resolve_prefers_uuid_then_exact_name(self):
        alice = Node(uuid="n1", name="Alice", labels=["PERSON"])
        other = Node(uuid="n2", name="Alicia", labels=["PERSON"])
        other_hit = MagicMock()
        other_hit.metadata = {"uuid": "n2"}

        def fake_text_search(text, brain_id="default"):
            if brain_id == "miss":
                return [other]
            return [alice, other]

        with (
            patch(
                "src.core.search.entity_resolve.graph_adapter.get_by_uuid",
                side_effect=lambda uuid, brain_id="default": {
                    "n1": alice,
                    "n2": other,
                }.get(uuid),
            ),
            patch(
                "src.core.search.entity_resolve.graph_adapter.node_text_search",
                side_effect=fake_text_search,
            ),
            patch(
                "src.core.search.entity_resolve.embeddings_adapter.embed_text",
                side_effect=RuntimeError("embedder down"),
            ),
            patch(
                "src.core.search.entity_resolve.vector_search.search_nodes",
                return_value=[other_hit],
            ),
        ):
            self.assertEqual(resolve_entity_node("n1", "brain").uuid, "n1")
            self.assertEqual(resolve_entity_node("Alice", "brain").uuid, "n1")
            self.assertIsNone(resolve_entity_node("Alice", "miss"))

    def test_retrieve_matches_does_not_invent_empty_predicate(self):
        retriever = EventSynergyRetriever("brain")
        with patch(
            "src.core.search.entity_info.resolve_entity_node",
            return_value=None,
        ):
            path = retriever.retrieve_matches("missing", "query")
        self.assertIsInstance(path, MatchPath)
        self.assertIsNone(path.target_node)
        self.assertIsNone(path.path)


class RetrieveOffloadTests(unittest.TestCase):
    def test_extract_elements_is_offloaded(self):
        from src.services.api.controllers import retrieve as retrieve_mod

        source = inspect.getsource(retrieve_mod._build_context)
        self.assertIn("asyncio.to_thread(_extract_elements)", source)
        self.assertIn("asyncio.to_thread(_resolve_history_flag)", source)
        self.assertIn("asyncio.to_thread(_bridge_expand)", source)
        self.assertIn("asyncio.to_thread(_adjacency)", source)
        self.assertIn('profile_stage("ppr", blocking=False)', source)


class EntityInfoControllerTests(unittest.IsolatedAsyncioTestCase):
    async def test_missing_entity_returns_404(self):
        from fastapi import HTTPException
        from src.services.api.controllers import entities as entities_mod

        missing = MatchPath(target_node=None, path=None, similarity=0.0, children=[])
        retriever = MagicMock()
        retriever.retrieve_matches.return_value = missing
        with patch.object(entities_mod, "EventSynergyRetriever", return_value=retriever):
            with self.assertRaises(HTTPException) as ctx:
                await entities_mod.get_entity_info("nobody", "query")
        self.assertEqual(ctx.exception.status_code, 404)


class VectorStoreControllerTests(unittest.IsolatedAsyncioTestCase):
    async def test_unknown_store_returns_404(self):
        from fastapi import HTTPException
        from src.services.api.controllers import vectors as vectors_mod

        with self.assertRaises(HTTPException) as ctx:
            await vectors_mod.list_vectors("not-a-store", "brain")
        self.assertEqual(ctx.exception.status_code, 404)


class ModelEmbedTests(unittest.IsolatedAsyncioTestCase):
    async def test_add_nodes_embeds_and_stores_vector_id(self):
        from src.services.api.controllers import model as model_mod

        created = []

        def fake_add_nodes(nodes, brain_id, identification_params=None, metadata=None):
            created.extend(nodes)
            return nodes

        embed = MagicMock()
        embed.embed_text.return_value = MagicMock(embeddings=[0.1, 0.2], metadata={})
        store = MagicMock()
        store.add_vectors.return_value = ["vec-1"]
        with (
            patch.object(model_mod, "graph_adapter") as graph,
            patch.object(model_mod, "embeddings_adapter", embed),
            patch.object(model_mod, "vector_store_adapter", store),
        ):
            graph.add_nodes.side_effect = fake_add_nodes
            result = await model_mod.add_nodes(
                [{"name": "Alice", "labels": ["PERSON"]}],
                "brain",
            )
        self.assertEqual(result[0].properties["v_id"], "vec-1")
        store.add_vectors.assert_called_once()
        self.assertEqual(store.add_vectors.call_args.kwargs["store"], "nodes")

    async def test_invalid_labels_raise_identifier_error(self):
        from src.services.api.controllers import model as model_mod

        with self.assertRaises(InvalidCypherIdentifier):
            await model_mod.add_nodes(
                [{"name": "Alice", "labels": ["PERSON;DROP"]}],
                "brain",
            )

    async def test_normal_labels_are_kept_without_uppercasing(self):
        from src.services.api.controllers import model as model_mod

        captured = []

        def fake_add_nodes(nodes, brain_id, identification_params=None, metadata=None):
            captured.extend(nodes)
            return nodes

        with (
            patch.object(model_mod, "graph_adapter") as graph,
            patch.object(model_mod, "embeddings_adapter") as embed,
            patch.object(model_mod, "vector_store_adapter"),
        ):
            graph.add_nodes.side_effect = fake_add_nodes
            embed.embed_text.side_effect = RuntimeError("embedder down")
            await model_mod.add_nodes(
                [
                    {
                        "name": "Alice",
                        "labels": ["Software Engineer", "foo-bar", "Café"],
                    }
                ],
                "brain",
            )
        self.assertEqual(
            captured[0].labels,
            ["Software Engineer", "foo-bar", "Café"],
        )

    async def test_update_node_empty_labels_does_not_fail(self):
        from src.constants.kg import Node as KgNode
        from src.services.api.controllers import model as model_mod

        with patch.object(model_mod, "graph_adapter") as graph:
            graph.update_node.return_value = KgNode(
                uuid="n1", name="Alice", labels=[]
            )
            result = await model_mod.update_node("n1", "brain", new_labels=[])
        self.assertEqual(result.uuid, "n1")
        graph.update_node.assert_called_once()
        self.assertEqual(graph.update_node.call_args.kwargs["new_labels"], [])

    async def test_embed_failure_still_writes_node(self):
        from src.services.api.controllers import model as model_mod

        created = []

        def fake_add_nodes(nodes, brain_id, identification_params=None, metadata=None):
            created.extend(nodes)
            return nodes

        embed = MagicMock()
        embed.embed_text.side_effect = RuntimeError("embedder down")
        store = MagicMock()
        with (
            patch.object(model_mod, "graph_adapter") as graph,
            patch.object(model_mod, "embeddings_adapter", embed),
            patch.object(model_mod, "vector_store_adapter", store),
        ):
            graph.add_nodes.side_effect = fake_add_nodes
            result = await model_mod.add_nodes(
                [{"name": "Alice", "labels": ["PERSON"]}],
                "brain",
            )
        self.assertEqual(result[0].name, "Alice")
        self.assertNotIn("v_id", result[0].properties or {})
        store.add_vectors.assert_not_called()


if __name__ == "__main__":
    unittest.main()
