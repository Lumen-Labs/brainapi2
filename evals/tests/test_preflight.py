from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from evals.config import Settings
from evals.evaluate import evaluate
from evals.preflight import (
    collect_reported_chat_models,
    model_ids_match,
    preflight_brainapi_model,
)


class PreflightTests(unittest.TestCase):
    def test_unset_expected_does_not_enforce_a_model(self) -> None:
        settings = Settings.load()
        settings.expected_brainapi_model = ""
        result = preflight_brainapi_model(
            settings,
            payloads={
                "/health": {
                    "status": "ok",
                    "small_llm_model": "claude-sonnet-5-5",
                    "large_llm_model": "gpt-4o",
                    "embedding_model": "text-embedding-3-large",
                }
            },
        )
        self.assertEqual(result["mode"], "unchecked")
        self.assertEqual(result["llm_model"], settings.llm_model)

    def test_expected_model_must_match_report(self) -> None:
        settings = Settings.load()
        settings.expected_brainapi_model = "gpt-4o"
        with self.assertRaises(SystemExit) as ctx:
            preflight_brainapi_model(
                settings,
                payloads={"/health": {"large_llm_model": "claude-sonnet-5-5"}},
            )
        self.assertIn("expected chat model", str(ctx.exception))
        self.assertIn("gpt-4o", str(ctx.exception))

        result = preflight_brainapi_model(
            settings,
            payloads={"/health": {"large_llm_model": "openai/gpt-4o"}},
        )
        self.assertEqual(result["mode"], "reported")

    def test_expected_model_fails_when_health_omits_chat(self) -> None:
        settings = Settings.load()
        settings.expected_brainapi_model = "gpt-4o"
        with self.assertRaises(SystemExit) as ctx:
            preflight_brainapi_model(
                settings,
                payloads={"/health": {"status": "ok", "service": "brainapi"}},
            )
        self.assertIn("EVAL_EXPECTED_BRAINAPI_MODEL", str(ctx.exception))

        settings.expected_brainapi_model = ""
        result = preflight_brainapi_model(
            settings,
            payloads={"/health": {"status": "ok", "service": "brainapi"}},
        )
        self.assertEqual(result["mode"], "unchecked")

    def test_model_id_match_strips_provider_prefix(self) -> None:
        self.assertTrue(model_ids_match("anthropic/claude-haiku-5-5", "claude-haiku-5-5"))
        self.assertFalse(model_ids_match("gpt-4o", "gpt-4o-mini"))

    def test_collect_ignores_embeddings(self) -> None:
        rows = collect_reported_chat_models(
            {
                "/health": {
                    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
                    "reranker": "cross-encoder/ms-marco-MiniLM-L-6-v2",
                }
            }
        )
        self.assertEqual(rows, [])

    def test_non_smoke_brainapi_rejects_skip_enrichment(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = Settings.load()
            settings.runs_dir = Path(tmp)
            settings.brainpat_token = "token"
            with self.assertRaises(SystemExit) as ctx:
                evaluate(
                    settings,
                    use_cases=["expert_knowhow"],
                    split="test",
                    systems=["brainapi"],
                    run_id="should-not-run",
                    answerer="extractive",
                    tune=False,
                    smoke=False,
                    use_judge=False,
                    skip_enrichment=True,
                )
            self.assertIn("CI-smoke only", str(ctx.exception))
            self.assertFalse((Path(tmp) / "should-not-run").exists())
