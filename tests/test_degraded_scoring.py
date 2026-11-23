import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS = ROOT / "benchmarks"
if str(BENCHMARKS) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS))

from degraded import (  # noqa: E402
    DEFAULT_MAX_DEGRADED_RATE,
    report_exit_code,
    score_degraded_rows,
    zero_degraded_scores,
)


class DegradedHelperTests(unittest.TestCase):
    def test_strict_accuracy_and_rate_gate(self):
        rows = [
            {"judge_correct": True, "degraded": []},
            {"judge_correct": True, "degraded": ["data.keyword_search"]},
        ]
        stats = score_degraded_rows(rows)
        self.assertEqual(stats["n_degraded"], 1)
        self.assertEqual(stats["degraded_rate"], 0.5)
        self.assertEqual(stats["strict_accuracy"], 0.5)
        self.assertEqual(stats["max_degraded_rate"], DEFAULT_MAX_DEGRADED_RATE)
        self.assertTrue(stats["exceeded_max_degraded_rate"])
        under = score_degraded_rows(rows, max_degraded_rate=0.5)
        self.assertFalse(under["exceeded_max_degraded_rate"])

    def test_zero_degraded_scores(self):
        rows = [
            {"llm_judge_score": 0.8, "tau_norm": 0.9, "degraded": []},
            {"llm_judge_score": 0.8, "tau_norm": 0.9, "degraded": ["vector.search_data"]},
        ]
        strict = zero_degraded_scores(rows)
        self.assertEqual(strict[0]["llm_judge_score"], 0.8)
        self.assertEqual(strict[1]["llm_judge_score"], 0.0)
        self.assertEqual(strict[1]["tau_norm"], 0.0)

    def test_report_exit_code(self):
        self.assertEqual(report_exit_code({"status": "ok"}), 0)
        self.assertEqual(report_exit_code({"status": "failed"}), 1)
        self.assertEqual(
            report_exit_code(
                {"status": "degraded", "integrity": {"exceeded_max_degraded_rate": True}}
            ),
            1,
        )


class LongMemEvalDegradedScoringTests(unittest.TestCase):
    def test_report_strict_accuracy(self):
        from longmemeval.report import build_report

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            rows = [
                {
                    "question_id": "q1",
                    "judge_correct": True,
                    "degraded": [],
                    "prediction": "yes",
                },
                {
                    "question_id": "q2",
                    "judge_correct": True,
                    "degraded": ["data.keyword_search"],
                    "prediction": "yes",
                },
            ]
            (run_dir / "answers.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            (run_dir / "ingest.jsonl").write_text("", encoding="utf-8")
            report = build_report(run_dir)
            self.assertEqual(report["metrics"]["headline_judge_accuracy"], 1.0)
            self.assertEqual(report["metrics"]["headline_judge_accuracy_strict"], 0.5)
            self.assertEqual(report["integrity"]["degraded_rows"], 1)
            self.assertEqual(report["integrity"]["degraded_rate"], 0.5)
            self.assertTrue(report["integrity"]["exceeded_max_degraded_rate"])
            self.assertEqual(report_exit_code(report), 1)


class BeamDegradedScoringTests(unittest.TestCase):
    def test_report_strict_score_counts_degraded_as_zero(self):
        from beam.config import ABILITY_NAMES
        from beam.report import build_report

        ability = ABILITY_NAMES[0]
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            rows = [
                {
                    "sample_id": "1",
                    "ability": ability,
                    "qa_index": 0,
                    "llm_judge_score": 1.0,
                    "degraded": [],
                    "prediction": "ok",
                },
                {
                    "sample_id": "1",
                    "ability": ability,
                    "qa_index": 1,
                    "llm_judge_score": 1.0,
                    "degraded": ["data.keyword_search"],
                    "prediction": "ok",
                },
            ]
            (run_dir / "answers.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            (run_dir / "ingest.jsonl").write_text("", encoding="utf-8")
            report = build_report(run_dir)
            self.assertEqual(report["metrics"]["headline_score"], 1.0)
            self.assertEqual(report["metrics"]["headline_score_strict"], 0.5)
            self.assertEqual(report["integrity"]["degraded_rows"], 1)
            self.assertTrue(report["integrity"]["exceeded_max_degraded_rate"])
            self.assertEqual(report_exit_code(report), 1)


if __name__ == "__main__":
    unittest.main()
