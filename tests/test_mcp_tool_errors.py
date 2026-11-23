import os
import unittest
from unittest.mock import patch

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

from src.services.mcp.errors import McpToolError, raise_if_tool_error_result
from src.services.mcp.main import (
    _search_memory_sync,
    _traverse_graph_sync,
    auth_token_var,
)


class McpToolErrorTests(unittest.TestCase):
    def test_error_strings_are_raised(self):
        with self.assertRaises(McpToolError):
            raise_if_tool_error_result("Unauthorized")
        with self.assertRaises(McpToolError):
            raise_if_tool_error_result("Error executing graph operation: boom")
        with self.assertRaises(McpToolError):
            raise_if_tool_error_result("Fact not found")
        self.assertEqual(raise_if_tool_error_result('{"ok": true}'), '{"ok": true}')

    def test_search_memory_unauthorized_raises(self):
        token = auth_token_var.set("bad-token")
        try:
            with patch("src.services.mcp.main.guard_brainpat", return_value=False):
                with self.assertRaises(McpToolError) as ctx:
                    _search_memory_sync("MATCH (n) RETURN n", "demo", False)
            self.assertEqual(str(ctx.exception), "Unauthorized")
        finally:
            auth_token_var.reset(token)

    def test_search_memory_graph_error_raises(self):
        token = auth_token_var.set("ok")
        try:
            with (
                patch("src.services.mcp.main.guard_brainpat", return_value=True),
                patch(
                    "src.services.mcp.main.graph_adapter.execute_operation",
                    return_value="Error executing graph operation: write rejected",
                ),
            ):
                with self.assertRaises(McpToolError) as ctx:
                    _search_memory_sync("DELETE n", "demo", False)
            self.assertIn("Error executing graph operation", str(ctx.exception))
        finally:
            auth_token_var.reset(token)

    def test_traverse_invalid_direction_raises(self):
        token = auth_token_var.set("ok")
        try:
            with patch("src.services.mcp.main.guard_brainpat", return_value=True):
                with self.assertRaises(McpToolError) as ctx:
                    _traverse_graph_sync(
                        "demo",
                        "n1",
                        "",
                        None,
                        2,
                        None,
                        None,
                        "sideways",
                        10,
                        False,
                    )
            self.assertIn("direction", str(ctx.exception))
        finally:
            auth_token_var.reset(token)

    def test_fastmcp_marks_raised_tool_errors(self):
        from mcp.types import CallToolResult, TextContent

        try:
            raise McpToolError("Unauthorized")
        except Exception as exc:
            result = CallToolResult(
                content=[TextContent(type="text", text=str(exc))],
                isError=True,
            )
        self.assertTrue(result.isError)
        self.assertEqual(result.content[0].text, "Unauthorized")


if __name__ == "__main__":
    unittest.main()
