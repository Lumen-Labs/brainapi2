from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from src.lib.llm.policy import (
    DEFAULT_LLM_MODEL,
    HAIKU_MODEL_ID,
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

    def test_is_haiku_accepts_id_and_dated_suffix(self):
        self.assertTrue(is_haiku_5_5("claude-haiku-5-5"))
        self.assertTrue(is_haiku_5_5("claude-haiku-5-5-20251015"))
        self.assertFalse(is_haiku_5_5("claude-3-5-haiku-latest"))
        self.assertFalse(is_haiku_5_5("claude-sonnet-4-20250514"))
        self.assertFalse(is_haiku_5_5("gpt-4o-mini"))

    def test_remote_defaults_are_anthropic_haiku(self):
        env = {"LLM_MODEL": "claude-haiku-5-5"}
        with patch.dict(os.environ, env, clear=False):
            for key in (
                "LLM_PROVIDER",
                "LLM_SMALL_PROVIDER",
                "LLM_LARGE_PROVIDER",
                "ANTHROPIC_SMALL_LLM_MODEL",
                "ANTHROPIC_LARGE_LLM_MODEL",
            ):
                os.environ.pop(key, None)
            self.assertEqual(default_provider_for_mode("remote"), "anthropic")
            self.assertEqual(resolve_role_provider("small", "remote"), "anthropic")
            self.assertEqual(resolve_role_provider("large", "remote"), "anthropic")
            rows = resolve_llm_roles("remote")
            self.assertEqual(
                rows,
                [
                    ("small", "anthropic", "claude-haiku-5-5"),
                    ("large", "anthropic", "claude-haiku-5-5"),
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
            "LLM_SMALL_PROVIDER": "anthropic",
            "LLM_LARGE_PROVIDER": "openai",
            "ANTHROPIC_SMALL_LLM_MODEL": "claude-haiku-5-5",
            "OPENAI_LARGE_LLM_MODEL": "gpt-4o",
        }
        with patch.dict(os.environ, env, clear=False):
            self.assertEqual(
                resolve_role_model("small", "anthropic"), "claude-haiku-5-5"
            )
            self.assertEqual(resolve_role_model("large", "openai"), "gpt-4o")
            rows = resolve_llm_roles("remote")
            self.assertEqual(rows[0], ("small", "anthropic", "claude-haiku-5-5"))
            self.assertEqual(rows[1], ("large", "openai", "gpt-4o"))

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
    def test_remote_config_routes_both_roles_to_haiku(self):
        env = {
            **_CONFIG_ENV,
            "MODELS_MODE": "remote",
            "LLM_MODEL": "claude-haiku-5-5",
            "ANTHROPIC_API_KEY": "test-anthropic-key",
            "EMBEDDINGS_PROVIDER": "openai",
            "OPENAI_API_KEY": "test-openai-key",
            "OPENAI_EMBEDDING_MODEL": "text-embedding-3-large",
        }
        with patch.dict(os.environ, env, clear=False):
            for key in (
                "LLM_PROVIDER",
                "LLM_SMALL_PROVIDER",
                "LLM_LARGE_PROVIDER",
                "ANTHROPIC_SMALL_LLM_MODEL",
                "ANTHROPIC_LARGE_LLM_MODEL",
            ):
                os.environ.pop(key, None)
            from src.config import Config

            cfg = Config()
            self.assertEqual(cfg.llm_model, "claude-haiku-5-5")
            self.assertEqual(cfg.llm_provider, "anthropic")
            self.assertEqual(cfg.llm_small_provider, "anthropic")
            self.assertEqual(cfg.llm_large_provider, "anthropic")
            self.assertEqual(cfg.anthropic.small_llm_model, "claude-haiku-5-5")
            self.assertEqual(cfg.anthropic.large_llm_model, "claude-haiku-5-5")

    def test_benchmark_defaults_are_haiku(self):
        from benchmarks.locomo.config import (
            DEFAULT_ANSWER_MODEL,
            DEFAULT_LLM_BASE_URL,
            DEFAULT_SOTA_ANSWER_MODEL,
        )

        self.assertEqual(DEFAULT_ANSWER_MODEL, "claude-haiku-5-5")
        self.assertEqual(DEFAULT_SOTA_ANSWER_MODEL, "claude-haiku-5-5")
        self.assertEqual(DEFAULT_LLM_BASE_URL, "https://api.anthropic.com/v1")

    def test_configured_model_reads_anthropic_roles(self):
        from src.core.saving import pipeline_runtime

        fake = SimpleNamespace(
            llm_small_provider="anthropic",
            llm_large_provider="anthropic",
            anthropic=SimpleNamespace(
                small_llm_model="claude-haiku-5-5",
                large_llm_model="claude-haiku-5-5",
            ),
            ollama=None,
        )
        with patch("src.core.saving.pipeline_runtime.config", fake, create=True):
            with patch("src.config.config", fake):
                self.assertEqual(
                    pipeline_runtime.configured_model("small"),
                    ("anthropic", "claude-haiku-5-5"),
                )
                self.assertEqual(
                    pipeline_runtime.configured_model("large"),
                    ("anthropic", "claude-haiku-5-5"),
                )


if __name__ == "__main__":
    unittest.main()
