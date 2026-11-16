from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from src.lib.llm.policy import (
    allowed_llm_models,
    configured_llm_model,
    default_provider_for_mode,
    log_and_enforce_llm_roles,
    resolve_llm_roles,
    resolve_role_model,
    resolve_role_provider,
)


_CONFIG_ENV = {
    "BRAINPAT_TOKEN": "test-token",
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
    "NEO4J_PASSWORD": "test-password",
    "MILVUS_HOST": "localhost",
    "MILVUS_PORT": "19530",
    "MONGO_CONNECTION_STRING": "mongodb://localhost:27017",
    "POSTGRES_HOST": "localhost",
    "POSTGRES_PORT": "5432",
    "POSTGRES_USERNAME": "postgres",
    "POSTGRES_PASSWORD": "test-password",
    "CELERY_WORKER_CONCURRENCY": "1",
}

_ROUTING_KEYS = (
    "LLM_MODEL",
    "LLM_PROVIDER",
    "LLM_SMALL_PROVIDER",
    "LLM_LARGE_PROVIDER",
    "LLM_ALLOWED_MODELS",
    "ANTHROPIC_SMALL_LLM_MODEL",
    "ANTHROPIC_LARGE_LLM_MODEL",
    "OPENAI_SMALL_LLM_MODEL",
    "OPENAI_LARGE_LLM_MODEL",
    "AZURE_SMALL_LLM_MODEL",
    "AZURE_LARGE_LLM_MODEL",
    "BEDROCK_SMALL_LLM_MODEL",
    "BEDROCK_LARGE_LLM_MODEL",
    "GCP_SMALL_LLM_MODEL",
    "GCP_LARGE_LLM_MODEL",
    "OLLAMA_LLM_SMALL_MODEL",
    "OLLAMA_LLM_LARGE_MODEL",
)


def _clear_routing_env() -> None:
    for key in _ROUTING_KEYS:
        os.environ.pop(key, None)


class LlmRoutingPolicyTests(unittest.TestCase):
    def test_library_defaults_match_main(self):
        with patch.dict(os.environ, {}, clear=False):
            _clear_routing_env()
            self.assertIsNone(configured_llm_model())
            self.assertEqual(allowed_llm_models(), [])
            self.assertEqual(default_provider_for_mode("local"), "ollama")
            self.assertEqual(default_provider_for_mode("remote", "small"), "gcp_vertex")
            self.assertEqual(default_provider_for_mode("remote", "large"), "azure")
            self.assertEqual(resolve_role_provider("small", "remote"), "gcp_vertex")
            self.assertEqual(resolve_role_provider("large", "remote"), "azure")
            self.assertIsNone(resolve_role_model("small", "amazon_bedrock"))
            self.assertIsNone(resolve_role_model("small", "gcp_vertex"))
            self.assertEqual(
                resolve_role_model("small", "anthropic"), "claude-3-5-haiku-latest"
            )
            self.assertEqual(
                resolve_role_model("large", "anthropic"), "claude-sonnet-4-20250514"
            )
            self.assertEqual(resolve_role_model("small", "openai"), "gpt-4o-mini")
            self.assertEqual(resolve_role_model("large", "openai"), "gpt-4o")
            self.assertEqual(resolve_role_model("small", "azure"), "gpt-4o-mini")
            self.assertIsNone(resolve_role_model("large", "azure"))

    def test_shared_llm_model_routes_every_provider(self):
        with patch.dict(os.environ, {"LLM_MODEL": "test-model-a"}, clear=False):
            for key in _ROUTING_KEYS:
                if key != "LLM_MODEL":
                    os.environ.pop(key, None)
            self.assertEqual(configured_llm_model(), "test-model-a")
            self.assertEqual(resolve_role_model("small", "openai"), "test-model-a")
            self.assertEqual(resolve_role_model("large", "openai"), "test-model-a")
            self.assertEqual(resolve_role_model("small", "azure"), "test-model-a")
            self.assertEqual(resolve_role_model("large", "azure"), "test-model-a")
            self.assertEqual(resolve_role_model("small", "anthropic"), "test-model-a")
            self.assertEqual(resolve_role_model("large", "anthropic"), "test-model-a")
            self.assertEqual(
                resolve_role_model("small", "amazon_bedrock"), "test-model-a"
            )
            self.assertEqual(resolve_role_model("small", "gcp_vertex"), "test-model-a")
            self.assertEqual(resolve_role_model("small", "deepseek"), "test-model-a")
            self.assertEqual(resolve_role_model("small", "ollama"), "test-model-a")

    def test_per_role_env_overrides_shared_model(self):
        env = {
            "LLM_MODEL": "shared-model",
            "LLM_SMALL_PROVIDER": "amazon_bedrock",
            "LLM_LARGE_PROVIDER": "anthropic",
            "BEDROCK_SMALL_LLM_MODEL": "bedrock-small",
            "ANTHROPIC_LARGE_LLM_MODEL": "anthropic-large",
        }
        with patch.dict(os.environ, env, clear=False):
            self.assertEqual(
                resolve_role_model("small", "amazon_bedrock"), "bedrock-small"
            )
            self.assertEqual(resolve_role_model("large", "anthropic"), "anthropic-large")
            rows = resolve_llm_roles("remote")
            self.assertEqual(rows[0], ("small", "amazon_bedrock", "bedrock-small"))
            self.assertEqual(rows[1], ("large", "anthropic", "anthropic-large"))

    def test_shared_provider_overrides_role_defaults(self):
        with patch.dict(os.environ, {"LLM_PROVIDER": "openai"}, clear=False):
            os.environ.pop("LLM_SMALL_PROVIDER", None)
            os.environ.pop("LLM_LARGE_PROVIDER", None)
            self.assertEqual(resolve_role_provider("small", "remote"), "openai")
            self.assertEqual(resolve_role_provider("large", "remote"), "openai")
        with patch.dict(
            os.environ,
            {"LLM_PROVIDER": "openai", "LLM_LARGE_PROVIDER": "azure"},
            clear=False,
        ):
            os.environ.pop("LLM_SMALL_PROVIDER", None)
            self.assertEqual(resolve_role_provider("small", "remote"), "openai")
            self.assertEqual(resolve_role_provider("large", "remote"), "azure")

    def test_allowed_models_empty_is_off(self):
        rows = [("small", "openai", "any-model"), ("large", "openai", "other-model")]
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("LLM_ALLOWED_MODELS", None)
            resolved = log_and_enforce_llm_roles(rows)
            self.assertEqual(resolved, rows)

    def test_allowed_models_rejects_other_ids(self):
        rows = [("small", "openai", "allowed-model"), ("large", "openai", "other-model")]
        with self.assertRaises(RuntimeError) as raised:
            log_and_enforce_llm_roles(rows, allowed=["allowed-model"])
        self.assertIn("LLM_ALLOWED_MODELS", str(raised.exception))
        self.assertIn("other-model", str(raised.exception))

    def test_remote_config_uses_shared_model_on_bedrock_without_keys(self):
        env = {
            **_CONFIG_ENV,
            "MODELS_MODE": "remote",
            "LLM_MODEL": "test-model-a",
            "LLM_SMALL_PROVIDER": "amazon_bedrock",
            "LLM_LARGE_PROVIDER": "amazon_bedrock",
            "BEDROCK_REGION": "us-west-2",
            "EMBEDDINGS_PROVIDER": "openai",
            "OPENAI_API_KEY": "test-key",
        }
        with patch.dict(os.environ, env, clear=False):
            for key in (
                "LLM_PROVIDER",
                "AWS_ACCESS_KEY_ID",
                "AWS_SECRET_ACCESS_KEY",
                "BEDROCK_ACCESS_KEY_ID",
                "BEDROCK_SECRET_ACCESS_KEY",
                "BEDROCK_SMALL_LLM_MODEL",
                "BEDROCK_LARGE_LLM_MODEL",
                "OPENAI_SMALL_LLM_MODEL",
                "OPENAI_LARGE_LLM_MODEL",
            ):
                os.environ.pop(key, None)
            from src.config import Config

            cfg = Config()
            self.assertEqual(cfg.llm_model, "test-model-a")
            self.assertIsNone(cfg.llm_provider)
            self.assertEqual(cfg.llm_small_provider, "amazon_bedrock")
            self.assertEqual(cfg.llm_large_provider, "amazon_bedrock")
            self.assertEqual(cfg.bedrock.small_llm_model, "test-model-a")
            self.assertEqual(cfg.bedrock.large_llm_model, "test-model-a")
            self.assertEqual(cfg.bedrock.region, "us-west-2")
            self.assertIsNone(cfg.bedrock.access_key_id)
            self.assertIsNone(cfg.bedrock.secret_access_key)

    def test_remote_config_keeps_main_provider_defaults(self):
        env = {
            **_CONFIG_ENV,
            "MODELS_MODE": "remote",
            "EMBEDDINGS_PROVIDER": "openai",
            "OPENAI_API_KEY": "test-key",
            "LLM_SMALL_PROVIDER": "openai",
            "LLM_LARGE_PROVIDER": "openai",
        }
        with patch.dict(os.environ, env, clear=False):
            _clear_routing_env()
            os.environ["LLM_SMALL_PROVIDER"] = "openai"
            os.environ["LLM_LARGE_PROVIDER"] = "openai"
            from src.config import Config

            cfg = Config()
            self.assertIsNone(cfg.llm_model)
            self.assertIsNone(cfg.llm_provider)
            self.assertEqual(cfg.llm_small_provider, "openai")
            self.assertEqual(cfg.llm_large_provider, "openai")
            self.assertEqual(cfg.openai.small_llm_model, "gpt-4o-mini")
            self.assertEqual(cfg.openai.large_llm_model, "gpt-4o")

    def test_configured_model_reads_ollama_and_bedrock(self):
        from src.core.saving.pipeline_runtime import configured_model

        fake = SimpleNamespace(
            llm_small_provider="amazon_bedrock",
            llm_large_provider="ollama",
            bedrock=SimpleNamespace(small_llm_model="bedrock-small"),
            ollama=SimpleNamespace(llm_large_model="ollama-large"),
        )
        with patch("src.core.saving.pipeline_runtime.config", fake, create=True):
            with patch("src.config.config", fake):
                self.assertEqual(
                    configured_model("small"), ("amazon_bedrock", "bedrock-small")
                )
                self.assertEqual(configured_model("large"), ("ollama", "ollama-large"))


if __name__ == "__main__":
    unittest.main()
