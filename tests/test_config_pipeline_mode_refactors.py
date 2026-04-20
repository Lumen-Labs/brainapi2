import os
from pathlib import Path
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
    "PIPELINE_MODE": "accurate",
    "OCR_MODE": "docling",
    "AGENTIC_ARCHITECTURE": "custom",
}
for key, value in ENV_DEFAULTS.items():
    os.environ.setdefault(key, value)

from src.config import Config
from src.core.pipeline import (
    AccuratePipelineModeStrategy,
    LightweightPipelineModeStrategy,
    PipelineModeStrategyFactory,
    resolve_pipeline_mode_strategy,
)


ROOT = Path(__file__).resolve().parent.parent


def read_source(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


class ConfigModeValidationTests(unittest.TestCase):
    def test_pipeline_mode_defaults_to_accurate(self):
        with patch.dict(os.environ, ENV_DEFAULTS, clear=False):
            os.environ.pop("PIPELINE_MODE", None)
            config = Config()
        self.assertEqual(config.pipeline_mode, "accurate")

    def test_invalid_pipeline_mode_raises_value_error(self):
        with patch.dict(os.environ, ENV_DEFAULTS, clear=False):
            os.environ["PIPELINE_MODE"] = "invalid-mode"
            with self.assertRaises(ValueError):
                Config()

    def test_invalid_ocr_mode_raises_value_error(self):
        with patch.dict(os.environ, ENV_DEFAULTS, clear=False):
            os.environ["OCR_MODE"] = "invalid-ocr"
            with self.assertRaises(ValueError):
                Config()

    def test_invalid_agentic_architecture_raises_value_error(self):
        with patch.dict(os.environ, ENV_DEFAULTS, clear=False):
            os.environ["AGENTIC_ARCHITECTURE"] = "invalid-arch"
            with self.assertRaises(ValueError):
                Config()


class PipelineModeStrategyFactoryTests(unittest.TestCase):
    def test_factory_builds_lightweight_strategy(self):
        strategy = PipelineModeStrategyFactory().create("lightweight")
        self.assertIsInstance(strategy, LightweightPipelineModeStrategy)
        self.assertFalse(strategy.should_extract_observations())
        self.assertEqual(strategy.scout_mode(), "coarse")

    def test_factory_builds_accurate_strategy(self):
        strategy = PipelineModeStrategyFactory().create("accurate")
        self.assertIsInstance(strategy, AccuratePipelineModeStrategy)
        self.assertTrue(strategy.should_extract_observations())
        self.assertIsNone(strategy.scout_mode())

    def test_resolver_rejects_unknown_mode(self):
        with self.assertRaises(ValueError):
            resolve_pipeline_mode_strategy("broken")


class PipelineStrategyIntegrationTests(unittest.TestCase):
    def test_ingestion_uses_pipeline_strategy_dispatch(self):
        source = read_source("src/workers/tasks/ingestion.py")
        self.assertIn(
            "pipeline_strategy = resolve_pipeline_mode_strategy(config.pipeline_mode)",
            source,
        )
        self.assertIn("pipeline_strategy.should_extract_observations()", source)

    def test_auto_kg_uses_pipeline_strategy_dispatch(self):
        source = read_source("src/core/saving/auto_kg.py")
        self.assertIn(
            "pipeline_strategy = resolve_pipeline_mode_strategy(config.pipeline_mode)",
            source,
        )
        self.assertIn("scout_mode = pipeline_strategy.scout_mode()", source)


if __name__ == "__main__":
    unittest.main()
