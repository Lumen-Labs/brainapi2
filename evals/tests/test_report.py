from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from evals.report import render_markdown, write_report


class ReportTests(unittest.TestCase):
    def test_render_includes_scale_and_no_claim(self) -> None:
        report = {
            "run_id": "local-smoke",
            "scale": "local/small-scale",
            "split": "smoke",
            "answerer": "extractive",
            "llm_model": "claude-haiku-5-5",
            "prompt_budget": {"max_passages": 8},
            "answer_prompt_sha256": "abc",
            "claim_policy": "No claim that BrainAPI beats hybrid search unless the numbers show it.",
            "env": {"brainapi_url": "http://localhost:8000"},
            "use_cases": [
                {
                    "use_case": "expert_knowhow",
                    "n_docs": 8,
                    "n_questions": 2,
                    "split": "smoke",
                    "comparison": {
                        "verdict": "incomplete",
                        "reason": "Both systems must complete the same items before a win is claimed.",
                    },
                    "systems": {
                        "hybrid_baseline": {
                            "metrics": {
                                "accuracy": 1.0,
                                "wilson_lo": 0.3,
                                "wilson_hi": 1.0,
                                "latency": {"p50_ms": 12, "p95_ms": 20},
                                "total_tokens": 0,
                                "llm_calls": 0,
                            },
                            "ingest": {"ingest_ms": 40},
                            "capabilities": {},
                            "tuning": {"tuned": False},
                        }
                    },
                }
            ],
            "judge_agreement": {"n": 8, "exact_match_kappa_vs_human": 0.5},
        }
        md = render_markdown(report)
        self.assertIn("local/small-scale", md)
        self.assertIn("No claim that BrainAPI beats hybrid search", md)
        self.assertIn("incomplete", md)
        self.assertIn("claude-haiku-5-5", md)
        with tempfile.TemporaryDirectory() as tmp:
            path = write_report(report, Path(tmp))
            self.assertTrue(path.exists())
            self.assertIn("expert_knowhow", path.read_text(encoding="utf-8"))
