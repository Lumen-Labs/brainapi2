from __future__ import annotations

import os
import unittest
from contextlib import contextmanager
from unittest.mock import patch

from evals.config import (
    DEFAULT_LLM_MODEL,
    Settings,
    canonicalize_llm_model,
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
    "ANTHROPIC_API_KEY",
    "DEEPSEEK_API_KEY",
    "OPENAI_API_KEY",
    "EVAL_BRAINAPI_LLM_CONFIRMED",
)


@contextmanager
def isolated_env(**values: str):
    with patch.dict(os.environ, values, clear=False):
        for key in _LLM_ENV:
            if key not in values:
                os.environ.pop(key, None)
        yield


class ConfigTests(unittest.TestCase):
    def test_default_llm_is_haiku(self) -> None:
        with isolated_env():
            settings = Settings.load()
        self.assertEqual(settings.llm_model, DEFAULT_LLM_MODEL)
        self.assertEqual(settings.answer_model, DEFAULT_LLM_MODEL)
        self.assertEqual(settings.judge_model, DEFAULT_LLM_MODEL)
        self.assertEqual(settings.llm_base_url, "https://api.anthropic.com/v1")
        self.assertEqual(settings.llm_api_key, "")

    def test_alias_normalizes_to_canonical_id(self) -> None:
        self.assertEqual(
            canonicalize_llm_model("claude-haiku-5.5", source="test"),
            DEFAULT_LLM_MODEL,
        )
        with isolated_env(EVAL_LLM_MODEL="anthropic/claude-haiku-5-5"):
            self.assertEqual(Settings.load().llm_model, DEFAULT_LLM_MODEL)

    def test_rejects_openai_and_deepseek_chat_models(self) -> None:
        for model in ("gpt-4o", "deepseek-v4-flash", "claude-sonnet-5-5"):
            with isolated_env(EVAL_LLM_MODEL=model):
                with self.assertRaises(SystemExit) as ctx:
                    Settings.load()
                self.assertIn("Haiku 5.5", str(ctx.exception))

    def test_rejects_legacy_answer_model_env(self) -> None:
        with isolated_env(EVAL_ANSWER_MODEL="deepseek-v4-flash"):
            with self.assertRaises(SystemExit):
                Settings.load()

    def test_rejects_deepseek_and_openai_chat_hosts(self) -> None:
        for url in (
            "https://api.deepseek.com",
            "https://api.openai.com/v1",
            "https://my.openai.azure.com/openai",
        ):
            with isolated_env(EVAL_LLM_BASE_URL=url):
                with self.assertRaises(SystemExit) as ctx:
                    Settings.load()
                self.assertIn("Haiku 5.5", str(ctx.exception))

    def test_rejects_azure_judge(self) -> None:
        with isolated_env(EVAL_JUDGE_AZURE_ENDPOINT="https://example.openai.azure.com"):
            with self.assertRaises(SystemExit):
                Settings.load()

    def test_chat_key_ignores_openai_and_deepseek(self) -> None:
        with isolated_env(
            OPENAI_API_KEY="sk-openai",
            DEEPSEEK_API_KEY="sk-deepseek",
            EVAL_LLM_API_KEY="sk-ant-eval",
        ):
            settings = Settings.load()
        self.assertEqual(settings.llm_api_key, "sk-ant-eval")

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

    def test_brainapi_confirmed_must_be_haiku(self) -> None:
        with isolated_env(EVAL_BRAINAPI_LLM_CONFIRMED="gpt-4o"):
            with self.assertRaises(SystemExit):
                Settings.load()
        with isolated_env(EVAL_BRAINAPI_LLM_CONFIRMED="claude-haiku-5-5"):
            settings = Settings.load()
        self.assertEqual(settings.brainapi_llm_confirmed, "claude-haiku-5-5")
