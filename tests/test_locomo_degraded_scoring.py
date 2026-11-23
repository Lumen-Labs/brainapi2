import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS = ROOT / "benchmarks"
if str(BENCHMARKS) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS))


def _answer_row(*, qa_index: int, correct: bool, degraded=None) -> dict:
    return {
        "sample_id": "conv-26",
        "qa_index": qa_index,
        "category": 1,
        "judge_correct": correct,
        "gold": "Alice",
        "prediction": "Alice" if correct else "Bob",
        "f1": 1.0 if correct else 0.0,
        "bleu1": 1.0 if correct else 0.0,
        "degraded": degraded or [],
        "text_context": "Alice is here",
        "graph_context": "Alice",
        "triples": [],
        "paths": [],
        "source_passages": [],
        "historical_context": [],
        "evidence": [],
    }


class LocomoDegradedScoringTests(unittest.TestCase):
    def test_extract_context_channels_records_degraded(self):
        from locomo.evaluate import extract_context_channels

        channels = extract_context_channels(
            {
                "text_context": "hello",
                "source_passages": [],
                "degraded": ["data.keyword_search", "vector.search_data"],
            }
        )
        self.assertEqual(
            channels["degraded"],
            ["data.keyword_search", "vector.search_data"],
        )

    def test_report_excludes_degraded_from_accuracy_by_default(self):
        from locomo.report import build_report, render_markdown

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            rows = [
                _answer_row(qa_index=0, correct=True),
                _answer_row(
                    qa_index=1,
                    correct=False,
                    degraded=["data.keyword_search"],
                ),
            ]
            (run_dir / "answers.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            (run_dir / "ingest.jsonl").write_text("", encoding="utf-8")
            report = build_report(run_dir)
            self.assertEqual(report["integrity"]["degraded_rows"], 1)
            self.assertEqual(report["integrity"]["scored_rows"], 1)
            self.assertEqual(report["metrics"]["n_non_adversarial"], 1)
            self.assertEqual(report["metrics"]["headline_judge_accuracy"], 1.0)
            self.assertIn(
                "1 rows had degraded retrieval and are excluded from accuracy scores.",
                report["warnings"],
            )
            markdown = render_markdown(report)
            self.assertIn("Degraded retrieval rows: 1", markdown)
            included = build_report(run_dir, include_degraded=True)
            self.assertEqual(included["integrity"]["scored_rows"], 2)
            self.assertEqual(included["metrics"]["n_non_adversarial"], 2)
            self.assertEqual(included["metrics"]["headline_judge_accuracy"], 0.5)


if __name__ == "__main__":
    unittest.main()
