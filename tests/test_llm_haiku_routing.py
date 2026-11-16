from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from src.lib.llm.policy import (
    DEFAULT_BEDROCK_HAIKU_MODEL,
    DEFAULT_LLM_MODEL,
    DEFAULT_REMOTE_PROVIDER,
    HAIKU_MODEL_ID,
    bedrock_model_id,
    configured_llm_model,
    default_provider_for_mode,
    is_haiku_5_5,
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


class HaikuPolicyTests(unittest.TestCase):
    def test_default_model_id(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("LLM_MODEL", None)
            self.assertEqual(configured_llm_model(), "claude-haiku-5-5")
            self.assertEqual(DEFAULT_LLM_MODEL, HAIKU_MODEL_ID)
            self.assertEqual(DEFAULT_REMOTE_PROVIDER, "amazon_bedrock")
            self.assertEqual(
                DEFAULT_BEDROCK_HAIKU_MODEL, "us.anthropic.claude-haiku-5-5"
            )

    def test_is_haiku_accepts_api_and_bedrock_ids(self):
        self.assertTrue(is_haiku_5_5("claude-haiku-5-5"))
        self.assertTrue(is_haiku_5_5("claude-haiku-5-5-20251015"))
        self.assertTrue(is_haiku_5_5("anthropic.claude-haiku-5-5"))
        self.assertTrue(is_haiku_5_5("us.anthropic.claude-haiku-5-5"))
        self.assertTrue(is_haiku_5_5("global.anthropic.claude-haiku-5-5"))
        self.assertTrue(is_haiku_5_5("eu.anthropic.claude-haiku-5-5"))
        self.assertFalse(is_haiku_5_5("claude-3-5-haiku-latest"))
        self.assertFalse(is_haiku_5_5("us.anthropic.claude-3-5-haiku-20241022-v1:0"))
        self.assertFalse(is_haiku_5_5("us.anthropic.claude-3-7-sonnet-20250219-v1:0"))
        self.assertFalse(is_haiku_5_5("gpt-4o-mini"))

    def test_bedrock_model_id_maps_haiku(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("BEDROCK_INFERENCE_PROFILE", None)
            self.assertEqual(
                bedrock_model_id("claude-haiku-5-5"),
                "us.anthropic.claude-haiku-5-5",
            )
            self.assertEqual(
                bedrock_model_id("us.anthropic.claude-haiku-5-5"),
                "us.anthropic.claude-haiku-5-5",
            )
        with patch.dict(
            os.environ, {"BEDROCK_INFERENCE_PROFILE": "global"}, clear=False
        ):
            self.assertEqual(
                bedrock_model_id("claude-haiku-5-5"),
                "global.anthropic.claude-haiku-5-5",
            )

    def test_remote_defaults_are_bedrock_haiku(self):
        env = {"LLM_MODEL": "claude-haiku-5-5"}
        with patch.dict(os.environ, env, clear=False):
            for key in (
                "LLM_PROVIDER",
                "LLM_SMALL_PROVIDER",
                "LLM_LARGE_PROVIDER",
                "BEDROCK_SMALL_LLM_MODEL",
                "BEDROCK_LARGE_LLM_MODEL",
                "BEDROCK_INFERENCE_PROFILE",
            ):
                os.environ.pop(key, None)
            self.assertEqual(default_provider_for_mode("remote"), "amazon_bedrock")
            self.assertEqual(resolve_role_provider("small", "remote"), "amazon_bedrock")
            self.assertEqual(resolve_role_provider("large", "remote"), "amazon_bedrock")
            rows = resolve_llm_roles("remote")
            self.assertEqual(
                rows,
                [
                    ("small", "amazon_bedrock", "us.anthropic.claude-haiku-5-5"),
                    ("large", "amazon_bedrock", "us.anthropic.claude-haiku-5-5"),
                ],
            )

    def test_local_mode_still_defaults_to_ollama(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("LLM_SMALL_PROVIDER", None)
            os.environ.pop("LLM_LARGE_PROVIDER", None)
            os.environ.pop("LLM_PROVIDER", None)
            self.assertEqual(default_provider_for_mode("local"), "ollama")
            self.assertEqual(resolve_role_provider("small", "local"), "ollama")

    def test_existing_per_role_overrides(self):
        env = {
            "LLM_MODEL": "claude-haiku-5-5",
            "LLM_SMALL_PROVIDER": "amazon_bedrock",
            "LLM_LARGE_PROVIDER": "anthropic",
            "BEDROCK_SMALL_LLM_MODEL": "us.anthropic.claude-haiku-5-5",
            "ANTHROPIC_LARGE_LLM_MODEL": "claude-haiku-5-5",
        }
        with patch.dict(os.environ, env, clear=False):
            self.assertEqual(
                resolve_role_model("small", "amazon_bedrock"),
                "us.anthropic.claude-haiku-5-5",
            )
            self.assertEqual(
                resolve_role_model("large", "anthropic"), "claude-haiku-5-5"
            )
            rows = resolve_llm_roles("remote")
            self.assertEqual(
                rows[0], ("small", "amazon_bedrock", "us.anthropic.claude-haiku-5-5")
            )
            self.assertEqual(rows[1], ("large", "anthropic", "claude-haiku-5-5"))

    def test_anthropic_role_models_default_to_llm_model(self):
        with patch.dict(os.environ, {"LLM_MODEL": "claude-haiku-5-5"}, clear=False):
            os.environ.pop("ANTHROPIC_SMALL_LLM_MODEL", None)
            os.environ.pop("ANTHROPIC_LARGE_LLM_MODEL", None)
            self.assertEqual(
                resolve_role_model("small", "anthropic"), "claude-haiku-5-5"
            )
            self.assertEqual(
                resolve_role_model("large", "anthropic"), "claude-haiku-5-5"
            )

    def test_strict_startup_fails_on_non_haiku(self):
        rows = [("small", "openai", "gpt-4o-mini"), ("large", "openai", "gpt-4o")]
        with self.assertRaises(RuntimeError) as raised:
            log_and_enforce_llm_roles(rows, strict=True)
        self.assertIn("gpt-4o-mini", str(raised.exception))
        self.assertIn("LLM_REQUIRE_HAIKU", str(raised.exception))

    def test_non_strict_startup_logs_but_allows_non_haiku(self):
        rows = [("small", "ollama", "small"), ("large", "ollama", "large")]
        resolved = log_and_enforce_llm_roles(rows, strict=False)
        self.assertEqual(resolved, rows)


class HaikuConfigResolutionTests(unittest.TestCase):
    def test_remote_config_routes_both_roles_to_bedrock_haiku(self):
        env = {
            **_CONFIG_ENV,
            "MODELS_MODE": "remote",
            "LLM_MODEL": "claude-haiku-5-5",
            "BEDROCK_REGION": "us-east-1",
            "EMBEDDINGS_PROVIDER": "openai",
            "OPENAI_API_KEY": "test-openai-key",
            "OPENAI_EMBEDDING_MODEL": "text-embedding-3-large",
        }
        with patch.dict(os.environ, env, clear=False):
            for key in (
                "LLM_PROVIDER",
                "LLM_SMALL_PROVIDER",
                "LLM_LARGE_PROVIDER",
                "BEDROCK_SMALL_LLM_MODEL",
                "BEDROCK_LARGE_LLM_MODEL",
                "BEDROCK_ACCESS_KEY_ID",
                "BEDROCK_SECRET_ACCESS_KEY",
                "AWS_ACCESS_KEY_ID",
                "AWS_SECRET_ACCESS_KEY",
                "BEDROCK_INFERENCE_PROFILE",
            ):
                os.environ.pop(key, None)
            from src.config import Config

            cfg = Config()
            self.assertEqual(cfg.llm_model, "claude-haiku-5-5")
            self.assertEqual(cfg.llm_provider, "amazon_bedrock")
            self.assertEqual(cfg.llm_small_provider, "amazon_bedrock")
            self.assertEqual(cfg.llm_large_provider, "amazon_bedrock")
            self.assertEqual(
                cfg.bedrock.small_llm_model, "us.anthropic.claude-haiku-5-5"
            )
            self.assertEqual(
                cfg.bedrock.large_llm_model, "us.anthropic.claude-haiku-5-5"
            )
            self.assertIsNone(cfg.bedrock.access_key_id)
            self.assertIsNone(cfg.anthropic)

    def test_benchmark_defaults_are_haiku(self):
        from benchmarks.locomo.config import (
            DEFAULT_ANSWER_MODEL,
            DEFAULT_SOTA_ANSWER_MODEL,
        )

        self.assertEqual(DEFAULT_ANSWER_MODEL, "claude-haiku-5-5")
        self.assertEqual(DEFAULT_SOTA_ANSWER_MODEL, "claude-haiku-5-5")

    def test_configured_model_reads_bedrock_roles(self):
        from src.core.saving import pipeline_runtime

        fake = SimpleNamespace(
            llm_small_provider="amazon_bedrock",
            llm_large_provider="amazon_bedrock",
            bedrock=SimpleNamespace(
                small_llm_model="us.anthropic.claude-haiku-5-5",
                large_llm_model="us.anthropic.claude-haiku-5-5",
            ),
            ollama=None,
        )
        with patch("src.core.saving.pipeline_runtime.config", fake, create=True):
            with patch("src.config.config", fake):
                self.assertEqual(
                    pipeline_runtime.configured_model("small"),
                    ("amazon_bedrock", "us.anthropic.claude-haiku-5-5"),
                )
                self.assertEqual(
                    pipeline_runtime.configured_model("large"),
                    ("amazon_bedrock", "us.anthropic.claude-haiku-5-5"),
                )


if __name__ == "__main__":
    unittest.main()
