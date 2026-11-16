from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from evals.config import Settings
from evals.evaluate import evaluate
from evals.report import write_report


class SmokeRunTests(unittest.TestCase):
    def test_hybrid_smoke_all_use_cases(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = Settings.load()
            settings.runs_dir = Path(tmp)
            settings.scale = "local/small-scale"
            report = evaluate(
                settings,
                use_cases=[
                    "versioned_rules",
                    "policy_conflicts",
                    "expert_knowhow",
                    "decision_precedent",
                ],
                split="smoke",
                systems=["hybrid"],
                run_id="ci-smoke",
                answerer="extractive",
                tune=True,
                smoke=True,
                use_judge=False,
                skip_enrichment=True,
            )
            self.assertEqual(report["scale"], "local/small-scale")
            self.assertFalse(report["brainapi_attempted"])
            self.assertEqual(len(report["use_cases"]), 4)
            self.assertEqual(report.get("baseline_tier"), "smoke_hash")
            for use in report["use_cases"]:
                hybrid = use["systems"]["hybrid_baseline"]
                self.assertIn("metrics", hybrid)
                self.assertGreater(hybrid["metrics"]["n"], 0)
                self.assertLessEqual(use["n_docs"], 20)
                self.assertIsNone(use["systems"].get("brainapi"))
                ingest_cfg = (hybrid.get("ingest") or {}).get("config") or {}
                self.assertEqual(ingest_cfg.get("dense"), "hash")
                self.assertEqual(ingest_cfg.get("reranker"), "lexical")
                if use["use_case"] == "versioned_rules":
                    self.assertGreater(hybrid["metrics"]["accuracy"] or 0, 0.0)
            md = write_report(report, Path(tmp) / "ci-smoke")
            self.assertTrue(md.exists())
            json_path = Path(tmp) / "ci-smoke" / "report.json"
            self.assertTrue(json_path.exists())

    def test_missing_token_skips_brainapi(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = Settings.load()
            settings.runs_dir = Path(tmp)
            settings.brainpat_token = ""
            os.environ.pop("BRAINPAT_TOKEN", None)
            report = evaluate(
                settings,
                use_cases=["expert_knowhow"],
                split="smoke",
                systems=["hybrid", "brainapi"],
                run_id="ci-skip",
                answerer="extractive",
                tune=False,
                smoke=True,
                use_judge=False,
                skip_enrichment=True,
            )
            self.assertFalse(report["brainapi_attempted"])
            self.assertIn("hybrid_baseline", report["use_cases"][0]["systems"])
