from __future__ import annotations

import os
import unittest
from contextlib import contextmanager
from unittest.mock import patch

from evals.config import (
    CHEAP_CROSS_ENCODER,
    CHEAP_ST_EMBEDDING_MODEL,
    DEFAULT_CONFLICT_LIST_PATH,
    DEFAULT_CONFLICT_SCAN_PATH,
    DEFAULT_CROSS_ENCODER,
    DEFAULT_ST_EMBEDDING_MODEL,
    Settings,
    embedding_dim_for,
    query_prefix_for_embedding_model,
)
from evals.llm import is_anthropic_api_host


_LLM_ENV = (
    "EVAL_LLM_MODEL",
    "EVAL_ANSWER_MODEL",
    "EVAL_JUDGE_MODEL",
    "BENCH_ANSWER_MODEL",
    "BENCH_JUDGE_MODEL",
    "EVAL_LLM_BASE_URL",
    "EVAL_JUDGE_BASE_URL",
    "BENCH_JUDGE_BASE_URL",
    "EVAL_JUDGE_AZURE_ENDPOINT",
    "EVAL_JUDGE_AZURE_KEY",
    "EVAL_LLM_API_KEY",
    "EVAL_LLM_PROVIDER",
    "ANTHROPIC_API_KEY",
    "DEEPSEEK_API_KEY",
    "OPENAI_API_KEY",
    "EVAL_BRAINAPI_LLM_CONFIRMED",
    "EVAL_EXPECTED_BRAINAPI_MODEL",
    "EVAL_BASELINE_PROFILE",
    "EVAL_EMBEDDING_MODEL",
    "EVAL_CROSS_ENCODER_MODEL",
    "EVAL_EMBEDDING_PROVIDER",
    "EVAL_CONFLICT_PATH",
    "EVAL_CONFLICT_SCAN_PATH",
)


@contextmanager
def isolated_env(**values: str):
    with patch.dict(os.environ, values, clear=False):
        for key in _LLM_ENV:
            if key not in values:
                os.environ[key] = ""
        yield


class ConfigTests(unittest.TestCase):
    def test_llm_model_is_plain_config(self) -> None:
        with isolated_env():
            settings = Settings.load()
        self.assertEqual(settings.llm_model, "")
        self.assertEqual(settings.answer_model, "")
        self.assertEqual(settings.judge_model, "")
        self.assertEqual(settings.llm_provider, "")
        self.assertIsNone(settings.llm_base_url)
        self.assertEqual(settings.llm_api_key, "")
        self.assertEqual(settings.expected_brainapi_model, "")

        with isolated_env(
            EVAL_LLM_MODEL="gpt-4o",
            EVAL_LLM_PROVIDER="openai",
            EVAL_LLM_BASE_URL="https://api.openai.com/v1",
        ):
            settings = Settings.load()
        self.assertEqual(settings.llm_model, "gpt-4o")
        self.assertEqual(settings.llm_provider, "openai")
        self.assertEqual(settings.llm_base_url, "https://api.openai.com/v1")

    def test_conflicting_model_envs_fail(self) -> None:
        with isolated_env(
            EVAL_LLM_MODEL="gpt-4o",
            EVAL_ANSWER_MODEL="deepseek-v4-flash",
        ):
            with self.assertRaises(SystemExit) as ctx:
                Settings.load()
        self.assertIn("Conflicting LLM model", str(ctx.exception))

    def test_legacy_answer_model_env_is_accepted(self) -> None:
        with isolated_env(EVAL_ANSWER_MODEL="deepseek-v4-flash"):
            settings = Settings.load()
        self.assertEqual(settings.llm_model, "deepseek-v4-flash")

    def test_openai_and_deepseek_hosts_are_allowed(self) -> None:
        for url in (
            "https://api.deepseek.com",
            "https://api.openai.com/v1",
            "https://my.openai.azure.com/openai",
        ):
            with isolated_env(EVAL_LLM_BASE_URL=url, EVAL_LLM_MODEL="any-model"):
                settings = Settings.load()
            self.assertEqual(settings.llm_base_url, url)

    def test_mismatched_judge_base_url_fails(self) -> None:
        with isolated_env(
            EVAL_LLM_BASE_URL="https://api.openai.com/v1",
            EVAL_JUDGE_BASE_URL="https://api.deepseek.com",
        ):
            with self.assertRaises(SystemExit) as ctx:
                Settings.load()
        self.assertIn("EVAL_JUDGE_BASE_URL", str(ctx.exception))

    def test_chat_key_prefers_eval_llm_key(self) -> None:
        with isolated_env(
            OPENAI_API_KEY="sk-openai",
            DEEPSEEK_API_KEY="sk-deepseek",
            EVAL_LLM_API_KEY="sk-eval",
        ):
            settings = Settings.load()
        self.assertEqual(settings.llm_api_key, "sk-eval")

    def test_openai_key_is_not_the_chat_key(self) -> None:
        with isolated_env(OPENAI_API_KEY="sk-openai"):
            settings = Settings.load()
        self.assertEqual(settings.llm_api_key, "")
        self.assertEqual(settings.embedding_api_key, "sk-openai")

    def test_anthropic_host_is_exact_or_subdomain(self) -> None:
        self.assertTrue(is_anthropic_api_host("api.anthropic.com"))
        self.assertTrue(is_anthropic_api_host("anthropic.com"))
        self.assertTrue(is_anthropic_api_host("API.Anthropic.COM."))
        self.assertFalse(is_anthropic_api_host("notanthropic.com"))
        self.assertFalse(is_anthropic_api_host("anthropic.com.evil.example"))
        self.assertFalse(is_anthropic_api_host(""))

    def test_expected_brainapi_model_is_plain_config(self) -> None:
        with isolated_env(EVAL_EXPECTED_BRAINAPI_MODEL="gpt-4o"):
            settings = Settings.load()
        self.assertEqual(settings.expected_brainapi_model, "gpt-4o")

    def test_require_llm_needs_model_and_key(self) -> None:
        with isolated_env():
            settings = Settings.load()
        with self.assertRaises(SystemExit) as ctx:
            settings.require_llm()
        self.assertIn("EVAL_LLM_MODEL", str(ctx.exception))
        settings.llm_model = "gpt-4o"
        with self.assertRaises(SystemExit) as ctx:
            settings.require_llm()
        self.assertIn("EVAL_LLM_API_KEY", str(ctx.exception))

    def test_default_baseline_is_bge_sota(self) -> None:
        with isolated_env():
            settings = Settings.load()
        self.assertEqual(settings.baseline_profile, "sota")
        self.assertEqual(settings.embedding_model, DEFAULT_ST_EMBEDDING_MODEL)
        self.assertEqual(settings.cross_encoder_model, DEFAULT_CROSS_ENCODER)

    def test_minilm_profile_selects_cheap_models(self) -> None:
        with isolated_env(EVAL_BASELINE_PROFILE="minilm"):
            settings = Settings.load()
        self.assertEqual(settings.baseline_profile, "minilm")
        self.assertEqual(settings.embedding_model, CHEAP_ST_EMBEDDING_MODEL)
        self.assertEqual(settings.cross_encoder_model, CHEAP_CROSS_ENCODER)

    def test_bge_en_gets_query_prefix_minilm_does_not(self) -> None:
        self.assertTrue(
            query_prefix_for_embedding_model("BAAI/bge-large-en-v1.5").startswith(
                "Represent this sentence"
            )
        )
        self.assertEqual(
            query_prefix_for_embedding_model("sentence-transformers/all-MiniLM-L6-v2"),
            "",
        )
        self.assertEqual(query_prefix_for_embedding_model("BAAI/bge-m3"), "")

    def test_conflict_paths_default_to_scan_then_list(self) -> None:
        with isolated_env():
            settings = Settings.load()
        self.assertEqual(settings.conflict_path, DEFAULT_CONFLICT_LIST_PATH)
        self.assertEqual(settings.conflict_scan_path, DEFAULT_CONFLICT_SCAN_PATH)

    def test_conflict_paths_can_be_disabled(self) -> None:
        with isolated_env(EVAL_CONFLICT_PATH="off", EVAL_CONFLICT_SCAN_PATH="none"):
            settings = Settings.load()
        self.assertIsNone(settings.conflict_path)
        self.assertIsNone(settings.conflict_scan_path)

    def test_embedding_dim_matches_profile_models(self) -> None:
        self.assertEqual(embedding_dim_for("BAAI/bge-large-en-v1.5"), 1024)
        self.assertEqual(
            embedding_dim_for("sentence-transformers/all-MiniLM-L6-v2"), 384
        )
