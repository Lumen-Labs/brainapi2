import os
import unittest
from unittest.mock import patch

from pydantic import BaseModel


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

from src.core.agents.architect_agent import ArchitectAgent
from src.core.agents.janitor_agent import JanitorAgent
from src.core.agents.kg_agent import KGAgent
from src.core.agents.tools.janitor_agent import (
    JanitorAgentExecuteGraphOperationTool,
    JanitorAgentExecuteGraphReadOperationTool,
)


class _DummyLLM:
    langchain_model = object()


class _DummyLLMAdapter:
    llm = _DummyLLM()


class _SchemaA(BaseModel):
    value_a: str


class _SchemaB(BaseModel):
    value_b: int


class _FakeToolStrategy:
    def __init__(self, schema):
        self.schema = schema


class KGAgentPromptStrategyTests(unittest.TestCase):
    def _build_agent(self):
        return KGAgent(
            llm_adapter=_DummyLLMAdapter(),
            cache_adapter=object(),
            kg=object(),
            vector_store=object(),
            embeddings=object(),
            database_desc="db",
        )

    @patch("src.core.agents.kg_agent.runtime_agent_factory.build")
    @patch("src.core.agents.kg_agent.prompt_registry.get")
    def test_get_agent_uses_registry_for_normal_prompt(self, get_prompt, build_agent):
        get_prompt.return_value = "normal::{extra_system_prompt}"
        kg_agent = self._build_agent()

        kg_agent._get_agent(
            type_="normal",
            identification_params={},
            metadata={},
            tools=[object()],
            extra_system_prompt="extra",
        )

        self.assertEqual(build_agent.call_args.kwargs["system_prompt"], "normal::extra")

    @patch("src.core.agents.kg_agent.runtime_agent_factory.build")
    @patch("src.core.agents.kg_agent.prompt_registry.get")
    def test_get_agent_uses_registry_for_graph_consolidator_prompt(
        self, get_prompt, build_agent
    ):
        get_prompt.return_value = "graph::{extra_system_prompt}"
        kg_agent = self._build_agent()

        kg_agent._get_agent(
            type_="graph-consolidator",
            identification_params={},
            metadata={},
            tools=[object()],
            extra_system_prompt="extra",
        )

        self.assertEqual(build_agent.call_args.kwargs["system_prompt"], "graph::extra")

    def test_get_agent_rejects_invalid_type(self):
        kg_agent = self._build_agent()
        with self.assertRaises(ValueError):
            kg_agent._get_agent(
                type_="invalid",  # type: ignore[arg-type]
                identification_params={},
                metadata={},
                tools=[object()],
            )


class JanitorAgentSchemaStrategyTests(unittest.TestCase):
    def _build_agent(self):
        return JanitorAgent(
            llm_adapter=_DummyLLMAdapter(),
            kg=object(),
            vector_store=object(),
            embeddings=object(),
            database_desc="db",
        )

    @patch("src.core.agents.janitor_agent.ToolStrategy", _FakeToolStrategy)
    def test_resolve_response_format_wraps_output_schemas(self):
        janitor_agent = self._build_agent()
        response_format = janitor_agent._resolve_response_format(
            output_schema=None, output_schemas=(_SchemaA, _SchemaB)
        )
        self.assertIsInstance(response_format, _FakeToolStrategy)
        self.assertIn("_SchemaA", str(response_format.schema))
        self.assertIn("_SchemaB", str(response_format.schema))

    @patch("src.core.agents.janitor_agent.ToolStrategy", _FakeToolStrategy)
    def test_resolve_response_format_wraps_tuple_output_schema(self):
        janitor_agent = self._build_agent()
        response_format = janitor_agent._resolve_response_format(
            output_schema=(_SchemaA, _SchemaB), output_schemas=None
        )
        self.assertIsInstance(response_format, _FakeToolStrategy)
        self.assertIn("_SchemaA", str(response_format.schema))
        self.assertIn("_SchemaB", str(response_format.schema))

    def test_resolve_response_format_keeps_single_schema_unwrapped(self):
        janitor_agent = self._build_agent()
        response_format = janitor_agent._resolve_response_format(
            output_schema=_SchemaA, output_schemas=None
        )
        self.assertIs(response_format, _SchemaA)

    def test_get_agent_rejects_invalid_type(self):
        janitor_agent = self._build_agent()
        with self.assertRaises(ValueError):
            janitor_agent._get_agent(type_="invalid")  # type: ignore[arg-type]


class ArchitectAgentPromptStrategyTests(unittest.TestCase):
    @patch("src.core.agents.architect_agent.prompt_registry.get")
    def test_resolve_system_prompt_uses_strategy_registry(self, get_prompt):
        get_prompt.return_value = "architect::{extra_system_prompt}"
        architect_agent = ArchitectAgent.__new__(ArchitectAgent)
        resolved = architect_agent._resolve_system_prompt(
            type_="tooler", mode="coarse", extra_system_prompt="extra"
        )
        self.assertEqual(resolved, "architect::extra")

    def test_resolve_system_prompt_rejects_invalid_mode(self):
        architect_agent = ArchitectAgent.__new__(ArchitectAgent)
        with self.assertRaises(ValueError):
            architect_agent._resolve_system_prompt(
                type_="tooler", mode="invalid"  # type: ignore[arg-type]
            )

    def test_resolve_system_prompt_rejects_invalid_type(self):
        architect_agent = ArchitectAgent.__new__(ArchitectAgent)
        with self.assertRaises(ValueError):
            architect_agent._resolve_system_prompt(
                type_="invalid", mode="granular"  # type: ignore[arg-type]
            )


class JanitorToolReuseTests(unittest.TestCase):
    def test_graph_operation_tool_reuses_read_operation_tool(self):
        self.assertTrue(
            issubclass(
                JanitorAgentExecuteGraphOperationTool,
                JanitorAgentExecuteGraphReadOperationTool,
            )
        )


if __name__ == "__main__":
    unittest.main()
