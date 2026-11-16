from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from evals.config import DEFAULT_LLM_MODEL, Settings
from evals.evaluate import evaluate
from evals.preflight import collect_reported_chat_models, preflight_brainapi_haiku


class PreflightTests(unittest.TestCase):
    def test_reported_haiku_passes_without_env(self) -> None:
        settings = Settings.load()
        settings.brainapi_llm_confirmed = ""
        result = preflight_brainapi_haiku(
            settings,
            payloads={
                "/health": {
                    "status": "ok",
                    "small_llm_model": "claude-haiku-5-5",
                    "large_llm_model": "claude-haiku-5-5",
                    "embedding_model": "text-embedding-3-large",
                }
            },
        )
        self.assertEqual(result["mode"], "reported")
        self.assertEqual(result["llm_model"], DEFAULT_LLM_MODEL)

    def test_reported_other_model_fails_even_if_confirmed(self) -> None:
        settings = Settings.load()
        settings.brainapi_llm_confirmed = DEFAULT_LLM_MODEL
        with self.assertRaises(SystemExit) as ctx:
            preflight_brainapi_haiku(
                settings,
                payloads={"/health": {"large_llm_model": "claude-sonnet-5-5"}},
            )
        self.assertIn("not Claude Haiku 5.5", str(ctx.exception))

    def test_missing_report_requires_confirmation(self) -> None:
        settings = Settings.load()
        settings.brainapi_llm_confirmed = ""
        with self.assertRaises(SystemExit) as ctx:
            preflight_brainapi_haiku(
                settings,
                payloads={"/health": {"status": "ok", "service": "brainapi"}},
            )
        self.assertIn("EVAL_BRAINAPI_LLM_CONFIRMED", str(ctx.exception))

        settings.brainapi_llm_confirmed = DEFAULT_LLM_MODEL
        result = preflight_brainapi_haiku(
            settings,
            payloads={"/health": {"status": "ok", "service": "brainapi"}},
        )
        self.assertEqual(result["mode"], "operator_confirmed")

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
